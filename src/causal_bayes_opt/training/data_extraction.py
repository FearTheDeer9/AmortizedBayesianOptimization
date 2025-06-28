#!/usr/bin/env python3
"""
Training Data Extraction from Expert Demonstrations

Extracts state-posterior pairs from expert trajectory demonstrations
for supervised training of the surrogate model.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as onp
import jax.numpy as jnp
import pyrsistent as pyr

from ..training.expert_collection.data_structures import (
    ExpertTrajectoryDemonstration,
    DemonstrationBatch
)
from ..data_structures.scm import get_variables, get_edges

logger = logging.getLogger(__name__)


# Import the existing TrainingExample from surrogate_training
from .surrogate_training import TrainingExample as SurrogateTrainingExample
from ..avici_integration.parent_set.posterior import ParentSetPosterior


@dataclass
class PosteriorHistoryExample:
    """Training example with posterior history metadata."""
    # Base training example
    training_example: SurrogateTrainingExample
    
    # Additional metadata for posterior history
    iteration: int
    trajectory_id: str
    
    def to_training_example(self) -> SurrogateTrainingExample:
        """Convert to standard training example."""
        return self.training_example


def extract_training_pairs_from_trajectory(
    trajectory_demo: ExpertTrajectoryDemonstration,
    trajectory_id: str
) -> List[PosteriorHistoryExample]:
    """
    Extract all state-posterior pairs from a single expert trajectory.
    
    Each intervention in the trajectory provides a training example where:
    - Input: Current data state (all observations + interventions so far)
    - Output: Posterior distribution over parent sets at that point
    
    Args:
        trajectory_demo: Expert trajectory demonstration with posterior history
        trajectory_id: Unique identifier for this trajectory
        
    Returns:
        List of training examples from this trajectory
    """
    training_examples = []
    
    # Extract trajectory and posterior history
    trajectory = trajectory_demo.expert_trajectory
    posterior_history = trajectory.get('posterior_history', [])
    
    if not posterior_history:
        logger.warning(f"No posterior history found in trajectory {trajectory_id}")
        return []
    
    # Get variable information
    scm = trajectory_demo.scm
    variable_order = sorted(list(scm.get('variables', frozenset())))
    target_variable = trajectory_demo.target_variable
    n_nodes = trajectory_demo.n_nodes
    graph_type = trajectory_demo.graph_type
    
    logger.info(f"Extracting {len(posterior_history)} training examples from trajectory {trajectory_id}")
    
    for state in posterior_history:
        iteration = state['iteration']
        posterior_dict = state['posterior']  # Dict mapping parent sets to probabilities
        data_state = state['data_samples']
        
        # Convert to ParentSetPosterior format
        parent_sets = []
        probabilities = []
        
        for parents, prob in posterior_dict.items():
            if isinstance(parents, tuple):
                parent_set = frozenset(parents)
            elif isinstance(parents, frozenset):
                parent_set = parents
            else:
                parent_set = frozenset()
            parent_sets.append(parent_set)
            probabilities.append(float(prob))
        
        # Create ParentSetPosterior object
        parent_set_probs = pyr.pmap({ps: prob for ps, prob in zip(parent_sets, probabilities)})
        uncertainty = -sum(p * jnp.log(p + 1e-10) for p in probabilities)  # Entropy
        top_k_sets = sorted(zip(parent_sets, probabilities), key=lambda x: x[1], reverse=True)[:5]
        
        expert_posterior = ParentSetPosterior(
            target_variable=target_variable,
            parent_set_probs=parent_set_probs,
            uncertainty=float(uncertainty),
            top_k_sets=top_k_sets
        )
        
        # Create placeholder observational data in AVICI format [N, d, 3]
        n_obs = data_state['observational']
        obs_data = jnp.zeros((n_obs, n_nodes, 3))
        # Set target indicator in channel 2
        target_idx = variable_order.index(target_variable)
        obs_data = obs_data.at[:, target_idx, 2].set(1.0)
        
        # Get expert accuracy (placeholder for now)
        expert_accuracy = 0.9  # Could be computed from trajectory metrics
        
        # Create SCM info
        scm_info = pyr.pmap({
            'n_nodes': n_nodes,
            'graph_type': graph_type,
            'edge_density': len(scm.get('edges', frozenset())) / (n_nodes * (n_nodes - 1))
        })
        
        # Create base TrainingExample
        base_example = SurrogateTrainingExample(
            observational_data=obs_data,
            target_variable=target_variable,
            variable_order=variable_order,
            expert_posterior=expert_posterior,
            expert_accuracy=expert_accuracy,
            scm_info=scm_info,
            problem_difficulty='medium'  # Could be inferred from convergence
        )
        
        # Wrap with history metadata
        example = PosteriorHistoryExample(
            training_example=base_example,
            iteration=iteration,
            trajectory_id=trajectory_id
        )
        
        training_examples.append(example)
    
    return training_examples


def extract_training_data_from_batch(
    batch: DemonstrationBatch,
    shuffle: bool = True,
    random_seed: int = 42
) -> List[SurrogateTrainingExample]:
    """
    Extract all training examples from a batch of demonstrations.
    
    Args:
        batch: Batch of expert demonstrations
        shuffle: Whether to shuffle the training examples
        random_seed: Random seed for shuffling
        
    Returns:
        List of all training examples from the batch
    """
    all_examples = []
    
    for i, demo in enumerate(batch.demonstrations):
        trajectory_id = f"{batch.batch_id}_demo_{i}"
        
        if isinstance(demo, ExpertTrajectoryDemonstration):
            # Handle trajectory demonstrations (with full trajectory)
            history_examples = extract_training_pairs_from_trajectory(demo, trajectory_id)
            # Convert to base training examples
            all_examples.extend([ex.to_training_example() for ex in history_examples])
        elif hasattr(demo, 'parent_posterior') and 'posterior_history' in demo.parent_posterior:
            # Handle regular demonstrations with posterior history
            history_examples = extract_training_pairs_from_expert_demonstration(demo, trajectory_id)
            # Convert to base training examples
            all_examples.extend([ex.to_training_example() for ex in history_examples])
        else:
            logger.warning(f"Skipping demonstration {i} - no posterior history found")
            continue
    
    logger.info(f"Extracted {len(all_examples)} total training examples from batch")
    
    # Shuffle if requested
    if shuffle and all_examples:
        rng = onp.random.RandomState(random_seed)
        indices = rng.permutation(len(all_examples))
        all_examples = [all_examples[i] for i in indices]
        logger.info("Shuffled training examples")
    
    return all_examples


def extract_training_pairs_from_expert_demonstration(
    demo: 'ExpertDemonstration',
    trajectory_id: str
) -> List[PosteriorHistoryExample]:
    """
    Extract training pairs from ExpertDemonstration with posterior history.
    
    Args:
        demo: Expert demonstration with posterior history
        trajectory_id: Unique identifier for this trajectory
        
    Returns:
        List of training examples from this demonstration
    """
    training_examples = []
    
    # Extract posterior history from the demonstration
    posterior_history = demo.parent_posterior.get('posterior_history', [])
    
    if not posterior_history:
        logger.warning(f"No posterior history found in demonstration {trajectory_id}")
        return []
    
    # Get variable information
    scm = demo.scm
    variable_order = sorted(list(get_variables(scm)))
    target_variable = demo.target_variable
    n_nodes = demo.n_nodes
    graph_type = demo.graph_type
    
    logger.info(f"Extracting {len(posterior_history)} training examples from demonstration {trajectory_id}")
    
    for state in posterior_history:
        iteration = state['iteration']
        posterior = state['posterior']  # Dict mapping parent sets to probabilities
        data_state = state['data_samples']
        
        # Convert posterior keys from tuples to frozensets
        parent_set_posterior = {}
        for parents, prob in posterior.items():
            if isinstance(parents, tuple):
                parent_set = frozenset(parents)
            elif isinstance(parents, frozenset):
                parent_set = parents
            else:
                parent_set = frozenset()
            parent_set_posterior[parent_set] = float(prob)
        
        # Convert to ParentSetPosterior format
        parent_sets = []
        probabilities = []
        
        for parent_set, prob in parent_set_posterior.items():
            parent_sets.append(parent_set)
            probabilities.append(float(prob))
        
        # Create ParentSetPosterior object
        parent_set_probs = pyr.pmap({ps: prob for ps, prob in zip(parent_sets, probabilities)})
        uncertainty = -sum(p * jnp.log(p + 1e-10) for p in probabilities)  # Entropy
        top_k_sets = sorted(zip(parent_sets, probabilities), key=lambda x: x[1], reverse=True)[:5]
        
        expert_posterior = ParentSetPosterior(
            target_variable=target_variable,
            parent_set_probs=parent_set_probs,
            uncertainty=float(uncertainty),
            top_k_sets=top_k_sets
        )
        
        # Create placeholder observational data in AVICI format [N, d, 3]
        n_obs = data_state['observational']
        obs_data = jnp.zeros((n_obs, n_nodes, 3))
        # Set target indicator in channel 2
        target_idx = variable_order.index(target_variable)
        obs_data = obs_data.at[:, target_idx, 2].set(1.0)
        
        # Get expert accuracy (placeholder for now)
        expert_accuracy = 0.9  # Could be computed from demonstration accuracy
        
        # Create SCM info
        scm_info = pyr.pmap({
            'n_nodes': n_nodes,
            'graph_type': graph_type,
            'edge_density': len(get_edges(scm)) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
        })
        
        # Create base TrainingExample
        base_example = SurrogateTrainingExample(
            observational_data=obs_data,
            target_variable=target_variable,
            variable_order=variable_order,
            expert_posterior=expert_posterior,
            expert_accuracy=expert_accuracy,
            scm_info=scm_info,
            problem_difficulty='medium'  # Could be inferred from convergence
        )
        
        # Wrap with history metadata
        example = PosteriorHistoryExample(
            training_example=base_example,
            iteration=iteration,
            trajectory_id=trajectory_id
        )
        
        training_examples.append(example)
    
    return training_examples


def format_for_surrogate_training(
    examples: List[SurrogateTrainingExample],
    max_parent_sets: int = 100
) -> Tuple[List[Dict[str, Any]], List[ParentSetPosterior]]:
    """
    Format training examples for surrogate model training.
    
    Args:
        examples: List of training examples
        max_parent_sets: Maximum number of parent sets to consider
        
    Returns:
        Tuple of (inputs, targets) ready for training
    """
    inputs = []
    targets = []
    
    for example in examples:
        # Format input
        input_dict = {
            'observational_data': example.observational_data,
            'target_variable': example.target_variable,
            'variable_order': example.variable_order,
            'scm_info': example.scm_info
        }
        inputs.append(input_dict)
        
        # Format target (posterior distribution)
        # The expert_posterior is already a ParentSetPosterior object
        # Optionally truncate to top k parent sets
        if len(example.expert_posterior.parent_sets) > max_parent_sets:
            # Get top k by probability
            probs = example.expert_posterior.probabilities
            top_k_indices = jnp.argsort(probs)[::-1][:max_parent_sets]
            
            top_k_parent_sets = [example.expert_posterior.parent_sets[i] for i in top_k_indices]
            top_k_probs = probs[top_k_indices]
            
            # Renormalize
            top_k_probs = top_k_probs / jnp.sum(top_k_probs)
            
            truncated_posterior = ParentSetPosterior(
                parent_sets=top_k_parent_sets,
                probabilities=top_k_probs
            )
            targets.append(truncated_posterior)
        else:
            targets.append(example.expert_posterior)
    
    return inputs, targets


def create_train_val_split(
    examples: List[SurrogateTrainingExample],
    val_fraction: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[SurrogateTrainingExample], List[SurrogateTrainingExample]]:
    """
    Split training examples into train and validation sets.
    
    Args:
        examples: All training examples
        val_fraction: Fraction of data for validation
        random_seed: Random seed for splitting
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    n_examples = len(examples)
    n_val = int(n_examples * val_fraction)
    
    # Shuffle and split
    rng = onp.random.RandomState(random_seed)
    indices = rng.permutation(n_examples)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    
    logger.info(f"Split data: {len(train_examples)} train, {len(val_examples)} validation")
    
    return train_examples, val_examples


def compute_training_statistics(examples: List[SurrogateTrainingExample]) -> Dict[str, Any]:
    """
    Compute statistics about the training data.
    
    Args:
        examples: List of training examples
        
    Returns:
        Dictionary of statistics
    """
    if not examples:
        return {}
    
    # Collect statistics
    n_examples = len(examples)
    graph_types = [ex.scm_info.get('graph_type', 'unknown') for ex in examples]
    node_sizes = [ex.scm_info.get('n_nodes', 0) for ex in examples]
    difficulties = [ex.problem_difficulty for ex in examples]
    accuracies = [ex.expert_accuracy for ex in examples]
    n_parent_sets = [len(ex.expert_posterior.parent_sets) for ex in examples]
    
    # Compute entropy of posterior distributions
    entropies = []
    for ex in examples:
        probs = ex.expert_posterior.probabilities
        probs = probs / jnp.sum(probs)  # Normalize
        entropy = -jnp.sum(probs * jnp.log(probs + 1e-10))
        entropies.append(float(entropy))
    
    stats = {
        'n_examples': n_examples,
        'graph_type_distribution': dict(zip(*onp.unique(graph_types, return_counts=True))) if graph_types else {},
        'node_size_distribution': dict(zip(*onp.unique(node_sizes, return_counts=True))) if node_sizes else {},
        'difficulty_distribution': dict(zip(*onp.unique(difficulties, return_counts=True))) if difficulties else {},
        'accuracy_stats': {
            'min': float(onp.min(accuracies)) if accuracies else 0,
            'max': float(onp.max(accuracies)) if accuracies else 0,
            'mean': float(onp.mean(accuracies)) if accuracies else 0,
            'std': float(onp.std(accuracies)) if accuracies else 0
        },
        'parent_set_stats': {
            'min': int(onp.min(n_parent_sets)) if n_parent_sets else 0,
            'max': int(onp.max(n_parent_sets)) if n_parent_sets else 0,
            'mean': float(onp.mean(n_parent_sets)) if n_parent_sets else 0,
            'std': float(onp.std(n_parent_sets)) if n_parent_sets else 0
        },
        'posterior_entropy_stats': {
            'min': float(onp.min(entropies)) if entropies else 0,
            'max': float(onp.max(entropies)) if entropies else 0,
            'mean': float(onp.mean(entropies)) if entropies else 0,
            'std': float(onp.std(entropies)) if entropies else 0
        }
    }
    
    return stats