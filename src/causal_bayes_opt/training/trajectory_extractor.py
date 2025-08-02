#!/usr/bin/env python3
"""
Trajectory Extractor for Behavioral Cloning

Pure functional trajectory processing following Rich Hickey's principles:
- Functional composition over large monolithic functions
- Immutable state transformations using pyrsistent
- JAX-native operations for performance
- Clear separation of concerns

This replaces the problematic trajectory_processor.py and bc_data_pipeline.py
"""

import logging
from typing import List, Tuple, Dict, FrozenSet, Optional, Callable, NamedTuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr
from pyrsistent import PRecord, field, PVector

from .pure_data_loader import DemonstrationData, AVICIData, PosteriorStep, InterventionStep
from ..avici_integration.parent_set.posterior import ParentSetPosterior
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState
from .data_structures import TrainingExample

logger = logging.getLogger(__name__)


# Immutable training data structures
class SurrogateTrainingData(PRecord):
    """Immutable surrogate training data"""
    training_examples: PVector[TrainingExample] = field(type=PVector)
    metadata: pyr.PMap = field(type=pyr.PMap)


class AcquisitionTrainingData(PRecord):
    """Immutable acquisition training data"""
    state_action_pairs: PVector[Tuple[Dict[str, object], Dict[str, object]]] = field(type=PVector)
    metadata: pyr.PMap = field(type=pyr.PMap)


class TrajectoryExtractionConfig(NamedTuple):
    """Configuration for trajectory extraction"""
    max_trajectory_length: int = 20
    min_posterior_entropy: float = 0.01
    use_all_steps: bool = True
    filter_invalid_actions: bool = True


class CurriculumLevel(NamedTuple):
    """Curriculum difficulty level specification"""
    name: str
    min_nodes: int
    max_nodes: int
    min_complexity: float
    max_complexity: float
    target_accuracy: float


# Predefined curriculum levels
CURRICULUM_LEVELS = [
    CurriculumLevel("easy", 3, 5, 0.0, 5.0, 0.7),
    CurriculumLevel("medium", 4, 8, 3.0, 8.0, 0.8),
    CurriculumLevel("hard", 6, 12, 6.0, 12.0, 0.85),
    CurriculumLevel("expert", 10, 20, 10.0, 20.0, 0.9)
]


# Pure helper functions
def _validate_demonstration_data(demo_data: DemonstrationData) -> None:
    """
    Pure function: Validate demonstration data for trajectory extraction.
    
    Args:
        demo_data: Demonstration to validate
        
    Raises:
        ValueError: If data is invalid for trajectory extraction
    """
    if not demo_data.avici_data.samples.size:
        raise ValueError(f"No AVICI samples in demonstration {demo_data.demo_id}")
    
    if not demo_data.posterior_history:
        raise ValueError(f"No posterior history in demonstration {demo_data.demo_id}")
    
    if not demo_data.intervention_sequence:
        raise ValueError(f"No intervention sequence in demonstration {demo_data.demo_id}")
    
    if len(demo_data.posterior_history) != len(demo_data.intervention_sequence):
        raise ValueError(f"Mismatch between posterior history and intervention sequence "
                        f"in demonstration {demo_data.demo_id}")


@jax.jit
def _compute_posterior_entropy(probabilities: jnp.ndarray) -> float:
    """JAX-optimized posterior entropy computation"""
    safe_probs = jnp.where(probabilities > 0, probabilities, 1e-10)
    return -jnp.sum(probabilities * jnp.log(safe_probs))


def _create_parent_set_posterior(
    posterior_step: PosteriorStep, 
    target_variable: str
) -> ParentSetPosterior:
    """
    Pure function: Create ParentSetPosterior from posterior step.
    
    Args:
        posterior_step: Posterior data for one step
        target_variable: Target variable for this posterior
        
    Returns:
        ParentSetPosterior object
    """
    parent_sets = list(posterior_step.posterior.keys())
    probabilities = list(posterior_step.posterior.values())
    
    if not parent_sets:
        # Empty posterior fallback
        parent_sets = [frozenset()]
        probabilities = [1.0]
    
    parent_set_probs = pyr.pmap(dict(zip(parent_sets, probabilities)))
    top_k_sets = sorted(
        zip(parent_sets, probabilities), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    return ParentSetPosterior(
        target_variable=target_variable,
        parent_set_probs=parent_set_probs,
        uncertainty=posterior_step.entropy,
        top_k_sets=top_k_sets
    )


def _create_acquisition_state(
    demo_data: DemonstrationData,
    step: int,
    intervention_history: List[InterventionStep]
) -> Dict[str, object]:
    """
    Pure function: Create state dictionary for a trajectory step.
    
    Note: We create a simplified state representation since the full AcquisitionState
    requires complex buffer and posterior structures that are not available in demos.
    
    Args:
        demo_data: Source demonstration data
        step: Current step in trajectory
        intervention_history: History of interventions up to this step
        
    Returns:
        Dictionary representing the state
    """
    # Get current posterior
    posterior_step = demo_data.posterior_history[step]
    
    # Create intervention history up to current step
    history_up_to_step = intervention_history[:step]
    
    return {
        'target_variable': demo_data.target_variable,
        'current_step': step,
        'posterior_entropy': posterior_step.entropy,
        'posterior_distribution': dict(posterior_step.posterior),
        'avici_data': demo_data.avici_data.samples,
        'intervention_history': [
            {
                'variables': list(int_step.variables),
                'values': int_step.values,
                'step': int_step.step
            }
            for int_step in history_up_to_step
        ],
        'scm_metadata': {
            'n_nodes': demo_data.n_nodes,
            'graph_type': demo_data.graph_type,
            'complexity_score': demo_data.complexity_score
        }
    }


def _create_expert_action(intervention_step: InterventionStep) -> Dict[str, object]:
    """
    Pure function: Create expert action from intervention step.
    
    Args:
        intervention_step: Expert intervention
        
    Returns:
        Dictionary representing expert action
    """
    return {
        'intervention_variables': intervention_step.variables,
        'intervention_values': intervention_step.values,
        'step': intervention_step.step,
        'action_type': 'intervention' if intervention_step.variables else 'observe'
    }


def extract_surrogate_training_example(
    demo_data: DemonstrationData,
    step: int,
    config: TrajectoryExtractionConfig
) -> Optional[TrainingExample]:
    """
    Pure function: Extract one surrogate training example from trajectory step.
    
    Args:
        demo_data: Source demonstration data
        step: Trajectory step to extract
        config: Extraction configuration
        
    Returns:
        TrainingExample or None if step should be filtered
    """
    try:
        # Validate step is within bounds
        if step >= len(demo_data.posterior_history):
            return None
        
        posterior_step = demo_data.posterior_history[step]
        
        # Filter by entropy if configured
        if posterior_step.entropy < config.min_posterior_entropy:
            return None
        
        # Create expert posterior
        expert_posterior = _create_parent_set_posterior(
            posterior_step, 
            demo_data.target_variable
        )
        
        # Create parent sets from posterior
        parent_sets = [frozenset(ps) for ps in expert_posterior.get('parent_sets', [])]
        expert_probs = expert_posterior.get('probabilities', jnp.array([]))
        
        return TrainingExample(
            observational_data=demo_data.avici_data.samples,
            expert_posterior=expert_posterior,
            target_variable=demo_data.target_variable,
            variable_order=list(demo_data.avici_data.variables),
            expert_accuracy=demo_data.accuracy,
            problem_difficulty='unknown',  # Will be set by curriculum manager
            parent_sets=parent_sets,
            expert_probs=expert_probs
        )
        
    except Exception as e:
        logger.warning(f"Failed to extract surrogate example at step {step} "
                      f"from demo {demo_data.demo_id}: {e}")
        return None


def extract_acquisition_training_pair(
    demo_data: DemonstrationData,
    step: int,
    config: TrajectoryExtractionConfig
) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    """
    Pure function: Extract one acquisition training pair from trajectory step.
    
    Args:
        demo_data: Source demonstration data
        step: Trajectory step to extract
        config: Extraction configuration
        
    Returns:
        (state, action) pair or None if step should be filtered
    """
    try:
        # Validate step bounds
        if step >= len(demo_data.intervention_sequence) or step >= len(demo_data.posterior_history):
            return None
        
        intervention_step = demo_data.intervention_sequence[step]
        
        # Filter invalid actions if configured
        if config.filter_invalid_actions and not intervention_step.variables:
            return None
        
        # Create state and action
        state = _create_acquisition_state(
            demo_data, 
            step, 
            list(demo_data.intervention_sequence)
        )
        action = _create_expert_action(intervention_step)
        
        return (state, action)
        
    except Exception as e:
        logger.warning(f"Failed to extract acquisition pair at step {step} "
                      f"from demo {demo_data.demo_id}: {e}")
        return None


def extract_surrogate_training_data(
    demonstration_data: List[DemonstrationData],
    config: TrajectoryExtractionConfig = TrajectoryExtractionConfig()
) -> SurrogateTrainingData:
    """
    Pure function: Extract all surrogate training examples from demonstrations.
    
    Args:
        demonstration_data: List of demonstrations to process
        config: Extraction configuration
        
    Returns:
        SurrogateTrainingData with all extracted examples
    """
    training_examples = []
    extraction_stats = {
        'total_demonstrations': len(demonstration_data),
        'total_steps_processed': 0,
        'valid_examples_extracted': 0,
        'filtered_steps': 0
    }
    
    for demo_data in demonstration_data:
        try:
            _validate_demonstration_data(demo_data)
            
            # Extract examples from each step
            max_steps = min(
                len(demo_data.posterior_history),
                config.max_trajectory_length
            )
            
            for step in range(max_steps):
                extraction_stats['total_steps_processed'] += 1
                
                example = extract_surrogate_training_example(demo_data, step, config)
                if example is not None:
                    training_examples.append(example)
                    extraction_stats['valid_examples_extracted'] += 1
                else:
                    extraction_stats['filtered_steps'] += 1
                    
        except ValueError as e:
            logger.warning(f"Skipping invalid demonstration {demo_data.demo_id}: {e}")
            continue
    
    metadata = pyr.pmap({
        'extraction_config': config._asdict(),
        'extraction_stats': extraction_stats,
        'total_examples': len(training_examples)
    })
    
    logger.info(f"Extracted {len(training_examples)} surrogate training examples "
                f"from {extraction_stats['total_steps_processed']} trajectory steps")
    
    return SurrogateTrainingData(
        training_examples=pyr.pvector(training_examples),
        metadata=metadata
    )


def extract_acquisition_training_data(
    demonstration_data: List[DemonstrationData],
    config: TrajectoryExtractionConfig = TrajectoryExtractionConfig()
) -> AcquisitionTrainingData:
    """
    Pure function: Extract all acquisition training pairs from demonstrations.
    
    Args:
        demonstration_data: List of demonstrations to process
        config: Extraction configuration
        
    Returns:
        AcquisitionTrainingData with all extracted pairs
    """
    training_pairs = []
    extraction_stats = {
        'total_demonstrations': len(demonstration_data),
        'total_steps_processed': 0,
        'valid_pairs_extracted': 0,
        'filtered_steps': 0
    }
    
    for demo_data in demonstration_data:
        try:
            _validate_demonstration_data(demo_data)
            
            # Extract pairs from each step
            max_steps = min(
                len(demo_data.intervention_sequence),
                config.max_trajectory_length
            )
            
            for step in range(max_steps):
                extraction_stats['total_steps_processed'] += 1
                
                pair = extract_acquisition_training_pair(demo_data, step, config)
                if pair is not None:
                    training_pairs.append(pair)
                    extraction_stats['valid_pairs_extracted'] += 1
                else:
                    extraction_stats['filtered_steps'] += 1
                    
        except ValueError as e:
            logger.warning(f"Skipping invalid demonstration {demo_data.demo_id}: {e}")
            continue
    
    metadata = pyr.pmap({
        'extraction_config': config._asdict(),
        'extraction_stats': extraction_stats,
        'total_pairs': len(training_pairs)
    })
    
    logger.info(f"Extracted {len(training_pairs)} acquisition training pairs "
                f"from {extraction_stats['total_steps_processed']} trajectory steps")
    
    return AcquisitionTrainingData(
        state_action_pairs=pyr.pvector(training_pairs),
        metadata=metadata
    )


def classify_demonstration_difficulty(
    demo_data: DemonstrationData,
    curriculum_levels: List[CurriculumLevel] = CURRICULUM_LEVELS
) -> str:
    """
    Pure function: Classify demonstration difficulty for curriculum learning.
    
    Args:
        demo_data: Demonstration to classify
        curriculum_levels: Available curriculum levels
        
    Returns:
        Name of the appropriate curriculum level
    """
    n_nodes = demo_data.n_nodes
    complexity = demo_data.complexity_score
    accuracy = demo_data.accuracy
    
    # Find the best matching curriculum level
    best_level = curriculum_levels[0]  # Default to easiest
    
    for level in curriculum_levels:
        # Check if demonstration fits this level's criteria
        nodes_match = level.min_nodes <= n_nodes <= level.max_nodes
        complexity_match = level.min_complexity <= complexity <= level.max_complexity
        accuracy_sufficient = accuracy >= level.target_accuracy
        
        if nodes_match and complexity_match and accuracy_sufficient:
            best_level = level
        elif nodes_match and complexity_match:
            # Accept even if accuracy is lower (expert might have failed)
            best_level = level
    
    return best_level.name


def organize_by_curriculum(
    demonstration_data: List[DemonstrationData],
    curriculum_levels: List[CurriculumLevel] = CURRICULUM_LEVELS
) -> Dict[str, List[DemonstrationData]]:
    """
    Pure function: Organize demonstrations by curriculum difficulty.
    
    Args:
        demonstration_data: Demonstrations to organize
        curriculum_levels: Available curriculum levels
        
    Returns:
        Dictionary mapping level names to demonstration lists
    """
    curriculum_groups = {level.name: [] for level in curriculum_levels}
    classification_stats = {level.name: 0 for level in curriculum_levels}
    
    for demo_data in demonstration_data:
        difficulty = classify_demonstration_difficulty(demo_data, curriculum_levels)
        curriculum_groups[difficulty].append(demo_data)
        classification_stats[difficulty] += 1
    
    logger.info(f"Curriculum classification: {classification_stats}")
    
    return curriculum_groups


def create_balanced_training_split(
    demonstration_data: List[DemonstrationData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[DemonstrationData], List[DemonstrationData], List[DemonstrationData]]:
    """
    Pure function: Create balanced train/val/test split preserving curriculum distribution.
    
    Args:
        demonstration_data: All demonstrations
        train_ratio: Fraction for training
        val_ratio: Fraction for validation  
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_data, val_data, test_data) tuple
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Organize by curriculum first
    curriculum_groups = organize_by_curriculum(demonstration_data)
    
    train_data = []
    val_data = []
    test_data = []
    
    # Create deterministic splits within each curriculum level
    rng = onp.random.RandomState(random_seed)
    
    for level_name, level_demos in curriculum_groups.items():
        if not level_demos:
            continue
            
        # Shuffle demonstrations deterministically
        level_demos_copy = level_demos.copy()
        rng.shuffle(level_demos_copy)
        
        n_demos = len(level_demos_copy)
        n_train = int(n_demos * train_ratio)
        n_val = int(n_demos * val_ratio)
        n_test = n_demos - n_train - n_val  # Remaining go to test
        
        train_data.extend(level_demos_copy[:n_train])
        val_data.extend(level_demos_copy[n_train:n_train + n_val])
        test_data.extend(level_demos_copy[n_train + n_val:])
    
    # Final shuffle to mix curriculum levels
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)
    
    logger.info(f"Created balanced split: {len(train_data)} train, "
                f"{len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data


# Composition functions for complete pipeline
def extract_complete_training_data(
    demonstration_data: List[DemonstrationData],
    config: TrajectoryExtractionConfig = TrajectoryExtractionConfig(),
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    random_seed: int = 42
) -> Dict[str, object]:
    """
    Pure function: Complete pipeline to extract all training data.
    
    Args:
        demonstration_data: Raw demonstration data
        config: Extraction configuration
        split_ratios: (train, val, test) ratios
        random_seed: Random seed for splits
        
    Returns:
        Dictionary with complete training data organized by split and type
    """
    # Create splits
    train_data, val_data, test_data = create_balanced_training_split(
        demonstration_data, *split_ratios, random_seed
    )
    
    # Extract training data for each split
    splits = {
        'train': train_data,
        'val': val_data, 
        'test': test_data
    }
    
    result = {}
    
    for split_name, split_data in splits.items():
        if not split_data:
            # Handle empty splits
            result[split_name] = {
                'surrogate': SurrogateTrainingData(
                    training_examples=pyr.pvector([]),
                    metadata=pyr.pmap({'empty_split': True})
                ),
                'acquisition': AcquisitionTrainingData(
                    state_action_pairs=pyr.pvector([]),
                    metadata=pyr.pmap({'empty_split': True})
                )
            }
            continue
        
        surrogate_data = extract_surrogate_training_data(split_data, config)
        acquisition_data = extract_acquisition_training_data(split_data, config)
        
        result[split_name] = {
            'surrogate': surrogate_data,
            'acquisition': acquisition_data
        }
    
    # Add global metadata
    result['metadata'] = {
        'total_demonstrations': len(demonstration_data),
        'config': config._asdict(),
        'split_ratios': split_ratios,
        'random_seed': random_seed,
        'curriculum_distribution': {
            level: len(demos) 
            for level, demos in organize_by_curriculum(demonstration_data).items()
        }
    }
    
    return result