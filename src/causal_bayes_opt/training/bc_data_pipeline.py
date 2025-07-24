#!/usr/bin/env python3
"""
Behavioral Cloning Data Pipeline

Orchestrates data extraction from expert demonstrations and preparation
for both surrogate and acquisition model training. Handles curriculum
ordering, data splitting, and batch creation.

Key Features:
1. Process all demonstration files from directory
2. Create curriculum-ordered datasets
3. Prepare JAX-ready training data
4. Handle memory-efficient batch processing
5. Support progressive difficulty training

Design Principles (Rich Hickey Approved):
- Pure functions with explicit data flow
- Immutable data structures throughout
- Clear separation of concerns
- Composable data transformations
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from .behavioral_cloning_adapter import load_all_demonstrations
from .trajectory_processor import (
    DifficultyLevel,
    TrainValTestSplit,
    SurrogateDataset,
    AcquisitionDataset,
    CurriculumLevel,
    create_curriculum_ordering,
    split_demonstrations,
    prepare_surrogate_dataset,
    prepare_acquisition_dataset,
    filter_by_difficulty,
    filter_acquisition_by_difficulty
)
from .expert_collection.data_structures import ExpertDemonstration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedDataset:
    """Complete processed dataset for behavioral cloning."""
    train_split: TrainValTestSplit
    surrogate_datasets: Dict[DifficultyLevel, SurrogateDataset]
    acquisition_datasets: Dict[DifficultyLevel, AcquisitionDataset]
    curriculum_levels: List[CurriculumLevel]
    statistics: Dict[str, Any]


@dataclass(frozen=True)
class DatasetStatistics:
    """Statistics about the processed dataset."""
    total_demonstrations: int
    total_surrogate_examples: int
    total_acquisition_steps: int
    difficulty_distribution: Dict[DifficultyLevel, int]
    complexity_stats: Dict[str, float]
    graph_type_distribution: Dict[str, int]
    node_count_distribution: Dict[str, int]


def process_all_demonstrations(
    demo_dir: str,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42,
    max_examples_per_demo: Optional[int] = None
) -> ProcessedDataset:
    """
    Pure function: Process all demonstrations from directory into training datasets.
    
    Args:
        demo_dir: Directory containing demonstration pickle files
        split_ratios: (train, validation, test) split ratios
        random_seed: Seed for reproducible splitting
        max_examples_per_demo: Maximum examples per demonstration
        
    Returns:
        ProcessedDataset with all prepared training data
    """
    logger.info(f"Processing demonstrations from {demo_dir}")
    
    # Load all demonstrations
    all_demos = load_all_demonstrations(demo_dir)
    
    if not all_demos:
        raise ValueError(f"No demonstrations found in {demo_dir}")
    
    logger.info(f"Loaded {len(all_demos)} demonstrations")
    
    # Create curriculum ordering
    curriculum_levels = create_curriculum_ordering(all_demos)
    logger.info(f"Created {len(curriculum_levels)} curriculum levels")
    
    # Split data into train/val/test
    data_split = split_demonstrations(
        demos=all_demos,
        ratios=split_ratios,
        random_seed=random_seed
    )
    
    logger.info(f"Data split: {len(data_split.train)} train, "
                f"{len(data_split.validation)} val, {len(data_split.test)} test")
    
    # Prepare datasets for each difficulty level using training data
    surrogate_datasets = {}
    acquisition_datasets = {}
    
    for curriculum_level in curriculum_levels:
        level = curriculum_level.level
        level_demos = [
            demo for demo in data_split.train
            if demo in curriculum_level.demonstrations
        ]
        
        if not level_demos:
            logger.warning(f"No training demonstrations for difficulty level {level}")
            continue
        
        # Prepare surrogate dataset for this level
        surrogate_dataset = prepare_surrogate_dataset(
            demos=level_demos,
            max_examples_per_demo=max_examples_per_demo
        )
        surrogate_datasets[level] = surrogate_dataset
        
        # Prepare acquisition dataset for this level
        acquisition_dataset = prepare_acquisition_dataset(
            demos=level_demos,
            max_steps_per_demo=max_examples_per_demo
        )
        acquisition_datasets[level] = acquisition_dataset
        
        logger.info(f"Level {level}: {len(surrogate_dataset.training_examples)} surrogate examples, "
                    f"{len(acquisition_dataset.trajectory_steps)} acquisition steps")
    
    # Compute statistics
    statistics = compute_dataset_statistics(
        all_demos=all_demos,
        surrogate_datasets=surrogate_datasets,
        acquisition_datasets=acquisition_datasets
    )
    
    return ProcessedDataset(
        train_split=data_split,
        surrogate_datasets=surrogate_datasets,
        acquisition_datasets=acquisition_datasets,
        curriculum_levels=curriculum_levels,
        statistics=statistics
    )


def compute_dataset_statistics(
    all_demos: List[ExpertDemonstration],
    surrogate_datasets: Dict[DifficultyLevel, SurrogateDataset],
    acquisition_datasets: Dict[DifficultyLevel, AcquisitionDataset]
) -> Dict[str, Any]:
    """
    Pure function: Compute comprehensive statistics about the dataset.
    
    Args:
        all_demos: All expert demonstrations
        surrogate_datasets: Surrogate datasets by difficulty
        acquisition_datasets: Acquisition datasets by difficulty
        
    Returns:
        Dictionary of dataset statistics
    """
    from .behavioral_cloning_adapter import compute_demonstration_complexity
    
    # Basic counts
    total_demos = len(all_demos)
    total_surrogate_examples = sum(
        len(dataset.training_examples) for dataset in surrogate_datasets.values()
    )
    total_acquisition_steps = sum(
        len(dataset.trajectory_steps) for dataset in acquisition_datasets.values()
    )
    
    # Difficulty distribution
    difficulty_dist = {}
    for level, dataset in surrogate_datasets.items():
        difficulty_dist[level] = len(dataset.training_examples)
    
    # Complexity statistics
    complexities = [compute_demonstration_complexity(demo) for demo in all_demos]
    complexity_stats = {
        'min': float(onp.min(complexities)),
        'max': float(onp.max(complexities)),
        'mean': float(onp.mean(complexities)),
        'std': float(onp.std(complexities))
    }
    
    # Graph type distribution
    graph_types = [demo.graph_type for demo in all_demos]
    graph_type_dist = {}
    for graph_type in set(graph_types):
        graph_type_dist[graph_type] = graph_types.count(graph_type)
    
    # Node count distribution
    node_counts = [demo.n_nodes for demo in all_demos]
    node_count_dist = {}
    for count in set(node_counts):
        node_count_dist[str(count)] = node_counts.count(count)
    
    return {
        'total_demonstrations': total_demos,
        'total_surrogate_examples': total_surrogate_examples,
        'total_acquisition_steps': total_acquisition_steps,
        'difficulty_distribution': difficulty_dist,
        'complexity_stats': complexity_stats,
        'graph_type_distribution': graph_type_dist,
        'node_count_distribution': node_count_dist
    }


def create_curriculum_batches(
    dataset: SurrogateDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[List[int]]:
    """
    Pure function: Create batches for curriculum training.
    
    Args:
        dataset: Surrogate dataset to batch
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        random_seed: Seed for reproducible shuffling
        
    Returns:
        List of batch indices
    """
    n_examples = len(dataset.training_examples)
    indices = list(range(n_examples))
    
    if shuffle:
        rng = onp.random.RandomState(random_seed)
        rng.shuffle(indices)
    
    batches = []
    for i in range(0, n_examples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append(batch_indices)
    
    return batches


def get_progressive_curriculum(
    processed_dataset: ProcessedDataset,
    start_level: DifficultyLevel = DifficultyLevel.EASY
) -> List[Tuple[DifficultyLevel, SurrogateDataset, AcquisitionDataset]]:
    """
    Pure function: Get progressive curriculum for training.
    
    Args:
        processed_dataset: Processed dataset with all difficulty levels
        start_level: Starting difficulty level
        
    Returns:
        List of (level, surrogate_dataset, acquisition_dataset) tuples in order
    """
    available_levels = sorted(
        processed_dataset.surrogate_datasets.keys(),
        key=lambda x: x.value
    )
    
    # Filter levels starting from start_level
    start_value = start_level.value
    curriculum_levels = [
        level for level in available_levels if level.value >= start_value
    ]
    
    curriculum = []
    for level in curriculum_levels:
        surrogate_dataset = processed_dataset.surrogate_datasets[level]
        acquisition_dataset = processed_dataset.acquisition_datasets[level]
        curriculum.append((level, surrogate_dataset, acquisition_dataset))
    
    return curriculum


def prepare_validation_datasets(
    processed_dataset: ProcessedDataset,
    target_levels: Optional[List[DifficultyLevel]] = None
) -> Tuple[SurrogateDataset, AcquisitionDataset]:
    """
    Pure function: Prepare validation datasets from validation split.
    
    Args:
        processed_dataset: Processed dataset with train/val split
        target_levels: Specific difficulty levels to include (all if None)
        
    Returns:
        Tuple of (validation_surrogate_dataset, validation_acquisition_dataset)
    """
    val_demos = processed_dataset.train_split.validation
    
    if not val_demos:
        raise ValueError("No validation demonstrations available")
    
    # Prepare datasets from validation demonstrations
    val_surrogate = prepare_surrogate_dataset(demos=val_demos)
    val_acquisition = prepare_acquisition_dataset(demos=val_demos)
    
    # Filter by target levels if specified
    if target_levels is not None:
        val_surrogate = filter_by_difficulty(val_surrogate, target_levels)
        val_acquisition = filter_acquisition_by_difficulty(val_acquisition, target_levels)
    
    return val_surrogate, val_acquisition


def prepare_test_datasets(
    processed_dataset: ProcessedDataset,
    target_levels: Optional[List[DifficultyLevel]] = None
) -> Tuple[SurrogateDataset, AcquisitionDataset]:
    """
    Pure function: Prepare test datasets from test split.
    
    Args:
        processed_dataset: Processed dataset with train/val/test split
        target_levels: Specific difficulty levels to include (all if None)
        
    Returns:
        Tuple of (test_surrogate_dataset, test_acquisition_dataset)
    """
    test_demos = processed_dataset.train_split.test
    
    if not test_demos:
        raise ValueError("No test demonstrations available")
    
    # Prepare datasets from test demonstrations
    test_surrogate = prepare_surrogate_dataset(demos=test_demos)
    test_acquisition = prepare_acquisition_dataset(demos=test_demos)
    
    # Filter by target levels if specified
    if target_levels is not None:
        test_surrogate = filter_by_difficulty(test_surrogate, target_levels)
        test_acquisition = filter_acquisition_by_difficulty(test_acquisition, target_levels)
    
    return test_surrogate, test_acquisition


def memory_efficient_batch_iterator(
    dataset: SurrogateDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42
):
    """
    Memory-efficient iterator over training batches.
    
    Yields batches without loading all data into memory at once.
    
    Args:
        dataset: Surrogate dataset to iterate over
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        random_seed: Seed for reproducible shuffling
        
    Yields:
        Batches of training examples
    """
    batches = create_curriculum_batches(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        random_seed=random_seed
    )
    
    for batch_indices in batches:
        batch_examples = [
            dataset.training_examples[i] for i in batch_indices
        ]
        yield batch_examples


def create_scm_aware_batches(
    dataset: SurrogateDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[List[int]]:
    """
    Pure function: Create batches grouped by SCM to ensure consistent dimensions.
    
    Each batch contains examples from the same SCM, ensuring all examples
    in a batch have the same number of variables. This is critical for
    models that process batches in parallel.
    
    Args:
        dataset: Surrogate dataset to batch
        batch_size: Maximum size of each batch
        shuffle: Whether to shuffle SCMs (not examples within SCM)
        random_seed: Seed for reproducible shuffling
        
    Returns:
        List of batch indices grouped by SCM
    """
    # Group examples by SCM ID or number of variables
    scm_groups = {}
    for i, example in enumerate(dataset.training_examples):
        # Use scm_id if available, otherwise group by number of variables
        if hasattr(example, 'scm_id'):
            scm_id = example.scm_id
        else:
            # Group by number of variables as a proxy for SCM
            # This ensures examples with same dimensionality are batched together
            n_vars = example.observational_data.shape[1]  # Shape is [N, d, 3]
            scm_id = f"nvars_{n_vars}"
        
        if scm_id not in scm_groups:
            scm_groups[scm_id] = []
        scm_groups[scm_id].append(i)
    
    # Get list of SCM IDs
    scm_ids = list(scm_groups.keys())
    
    # Shuffle SCM order if requested
    if shuffle:
        rng = onp.random.RandomState(random_seed)
        rng.shuffle(scm_ids)
    
    # Create batches within each SCM group
    batches = []
    for scm_id in scm_ids:
        indices = scm_groups[scm_id]
        
        # Shuffle examples within SCM if requested
        if shuffle:
            rng.shuffle(indices)
        
        # Create batches from this SCM's examples
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(batch_indices)
    
    logger.info(f"Created {len(batches)} SCM-aware batches from {len(scm_ids)} SCM groups")
    return batches


def scm_aware_batch_iterator(
    dataset: SurrogateDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42
):
    """
    Memory-efficient iterator over SCM-aware training batches.
    
    Yields batches where all examples come from the same SCM,
    ensuring consistent number of variables within each batch.
    
    Args:
        dataset: Surrogate dataset to iterate over
        batch_size: Maximum size of each batch
        shuffle: Whether to shuffle the data
        random_seed: Seed for reproducible shuffling
        
    Yields:
        Batches of training examples from the same SCM
    """
    batches = create_scm_aware_batches(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        random_seed=random_seed
    )
    
    for batch_indices in batches:
        batch_examples = [
            dataset.training_examples[i] for i in batch_indices
        ]
        
        # Verify all examples in batch have same number of variables
        n_vars_list = [len(ex.variable_order) for ex in batch_examples]
        if len(set(n_vars_list)) > 1:
            logger.warning(f"Batch contains mixed variable counts: {set(n_vars_list)}")
        
        yield batch_examples


def create_acquisition_scm_aware_batches(
    trajectory_steps: List['TrajectoryStep'],
    batch_size: int = 32,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[List[int]]:
    """
    Pure function: Create batches grouped by demonstration/SCM for acquisition training.
    
    Each batch contains trajectory steps from the same demonstration, ensuring all steps
    in a batch have the same SCM structure and variable dimensions. This is critical for
    preventing variable mismatch errors during acquisition policy training.
    
    Args:
        trajectory_steps: List of TrajectoryStep objects to batch
        batch_size: Maximum size of each batch
        shuffle: Whether to shuffle demonstrations and steps within demonstrations
        random_seed: Seed for reproducible shuffling
        
    Returns:
        List of batch indices grouped by demonstration/SCM
    """
    # Import TrajectoryStep type if needed
    from .trajectory_processor import TrajectoryStep
    
    # Group steps by demonstration ID
    demo_groups = {}
    for i, step in enumerate(trajectory_steps):
        demo_id = step.demonstration_id
        
        if demo_id not in demo_groups:
            demo_groups[demo_id] = []
        demo_groups[demo_id].append(i)
    
    # Also verify variable consistency within each group as a safety check
    for demo_id, indices in demo_groups.items():
        # Get variable info from first step in this demonstration
        first_step = trajectory_steps[indices[0]]
        first_state = first_step.state
        
        # Extract variable list from the first state
        first_vars = None
        if hasattr(first_state, 'metadata') and 'scm_info' in first_state.metadata:
            scm_info = first_state.metadata['scm_info']
            if isinstance(scm_info, dict) and 'variables' in scm_info:
                first_vars = list(scm_info['variables'])
        elif hasattr(first_state, 'posterior') and hasattr(first_state.posterior, 'variable_order'):
            first_vars = list(first_state.posterior.variable_order)
        
        if first_vars:
            n_vars = len(first_vars)
            logger.debug(f"Demo {demo_id}: {len(indices)} steps, {n_vars} variables ({first_vars[:3]}...)")
    
    # Get list of demonstration IDs
    demo_ids = list(demo_groups.keys())
    
    # Shuffle demonstration order if requested
    if shuffle:
        rng = onp.random.RandomState(random_seed)
        rng.shuffle(demo_ids)
    
    # Create batches within each demonstration group
    batches = []
    for demo_id in demo_ids:
        indices = demo_groups[demo_id]
        
        # Shuffle steps within demonstration if requested
        if shuffle:
            rng.shuffle(indices)
        
        # Create batches from this demonstration's steps
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batches.append(batch_indices)
    
    logger.info(f"Created {len(batches)} acquisition SCM-aware batches from {len(demo_ids)} demonstrations")
    
    # Log batch size distribution for debugging
    batch_sizes = [len(batch) for batch in batches]
    logger.info(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, "
                f"mean={onp.mean(batch_sizes):.1f}")
    
    return batches


def log_dataset_info(processed_dataset: ProcessedDataset) -> None:
    """
    Log comprehensive information about the processed dataset.
    
    Args:
        processed_dataset: Processed dataset to log information about
    """
    stats = processed_dataset.statistics
    
    logger.info("=== Dataset Statistics ===")
    logger.info(f"Total demonstrations: {stats['total_demonstrations']}")
    logger.info(f"Total surrogate examples: {stats['total_surrogate_examples']}")
    logger.info(f"Total acquisition steps: {stats['total_acquisition_steps']}")
    
    logger.info("=== Difficulty Distribution ===")
    for level, count in stats['difficulty_distribution'].items():
        logger.info(f"{level}: {count} examples")
    
    logger.info("=== Complexity Statistics ===")
    comp_stats = stats['complexity_stats']
    logger.info(f"Min: {comp_stats['min']:.2f}, Max: {comp_stats['max']:.2f}, "
                f"Mean: {comp_stats['mean']:.2f}, Std: {comp_stats['std']:.2f}")
    
    logger.info("=== Graph Type Distribution ===")
    for graph_type, count in stats['graph_type_distribution'].items():
        logger.info(f"{graph_type}: {count} demonstrations")
    
    logger.info("=== Node Count Distribution ===")
    for node_count, count in stats['node_count_distribution'].items():
        logger.info(f"{node_count} nodes: {count} demonstrations")