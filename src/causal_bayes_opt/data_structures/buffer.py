"""
Mutable experience buffer for storing observational and interventional data.

This module provides an efficient append-only buffer optimized for 
RL training with fast querying capabilities.
"""

# Standard library imports
import time
import logging
from typing import List, Tuple, Dict, Optional, FrozenSet, Any, Iterator, Set
from collections import defaultdict
from dataclasses import dataclass

# Third-party imports  
import pyrsistent as pyr

# Local imports
from .sample import (
    get_values, get_intervention_type, get_intervention_targets, 
    is_observational, is_interventional
)

# Type aliases
Sample = pyr.PMap
Intervention = pyr.PMap

logger = logging.getLogger(__name__)


@dataclass
class BufferStatistics:
    """Statistics about the buffer contents."""
    total_samples: int
    num_observations: int
    num_interventions: int
    unique_variables: int
    unique_intervention_types: int
    unique_intervention_targets: int
    creation_time: float
    last_update_time: float


class ExperienceBuffer:
    """
    Mutable, append-only buffer for storing intervention-outcome pairs.
    
    Optimized for:
    - Fast appends (O(1))
    - Efficient querying by variable sets
    - Batch processing for neural networks
    - Memory efficiency
    
    This class follows selective mutability principles - it's mutable for performance
    in training loops but provides immutable views for safety.
    """
    
    def __init__(self):
        """Initialize an empty experience buffer."""
        # Core storage - mutable for performance
        self._observations: List[Sample] = []
        self._interventions: List[Tuple[Intervention, Sample]] = []  # (intervention, outcome)
        
        # Posterior storage - parallel to samples for temporal consistency
        self._obs_posteriors: List[Optional[Dict[str, Any]]] = []  # Posteriors for each observation
        self._int_posteriors: List[Optional[Dict[str, Any]]] = []  # Posteriors for each intervention
        
        # Indexing for fast queries - updated on insert
        self._obs_by_variables: Dict[FrozenSet[str], List[int]] = defaultdict(list)
        self._int_by_targets: Dict[FrozenSet[str], List[int]] = defaultdict(list)
        self._int_by_type: Dict[str, List[int]] = defaultdict(list)
        
        # Metadata
        self._creation_time = time.time()
        self._last_update_time = self._creation_time
        self._variable_coverage: Set[str] = set()
        
        logger.debug("Initialized empty ExperienceBuffer")
    
    # Core operations
    def add_observation(self, sample: Sample, posterior: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an observational sample to the buffer with optional posterior.
        
        Args:
            sample: Observational sample to add
            posterior: Optional posterior distribution at this timestep
            
        Raises:
            ValueError: If sample is not observational
        """
        if not is_observational(sample):
            raise ValueError("Can only add observational samples via add_observation()")
        
        # Add to storage
        obs_index = len(self._observations)
        self._observations.append(sample)
        self._obs_posteriors.append(posterior)
        
        # Update indices
        variables = frozenset(get_values(sample).keys())
        self._obs_by_variables[variables].append(obs_index)
        
        # Update metadata
        self._variable_coverage.update(variables)
        self._last_update_time = time.time()
        
        logger.debug(f"Added observational sample {obs_index} with variables: {variables}, has_posterior: {posterior is not None}")
    
    def add_intervention(self, intervention: Intervention, outcome: Sample, 
                        posterior: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an intervention-outcome pair to the buffer with optional posterior.
        
        Args:
            intervention: Intervention specification
            outcome: Sample resulting from the intervention
            posterior: Optional posterior distribution that led to this intervention
            
        Raises:
            ValueError: If outcome is not interventional or inconsistent with intervention
        """
        if not is_interventional(outcome):
            raise ValueError("Outcome sample must be interventional")
        
        # Validate consistency between intervention and outcome
        outcome_targets = get_intervention_targets(outcome)
        # Note: intervention targets validation would need intervention spec format
        
        # Add to storage
        int_index = len(self._interventions)
        self._interventions.append((intervention, outcome))
        self._int_posteriors.append(posterior)
        
        # Update indices
        intervention_type = get_intervention_type(outcome)
        targets = get_intervention_targets(outcome)
        
        self._int_by_type[intervention_type].append(int_index)
        self._int_by_targets[targets].append(int_index)
        
        # Update metadata
        outcome_variables = frozenset(get_values(outcome).keys())
        self._variable_coverage.update(outcome_variables)
        self._last_update_time = time.time()
        
        logger.debug(f"Added intervention {int_index} of type '{intervention_type}' targeting {targets}, has_posterior: {posterior is not None}")
    
    # Query operations
    def get_observations(self) -> List[Sample]:
        """Get all observational samples (returns a copy for safety)."""
        return self._observations.copy()
    
    def get_interventions(self) -> List[Tuple[Intervention, Sample]]:
        """Get all intervention-outcome pairs (returns a copy for safety)."""
        return self._interventions.copy()
    
    def get_all_samples(self) -> List[Sample]:
        """Get all samples (observational + intervention outcomes) combined."""
        all_samples = self._observations.copy()
        all_samples.extend([outcome for _, outcome in self._interventions])
        return all_samples
    
    def get_all_samples_with_posteriors(self) -> List[Tuple[Sample, Optional[Dict[str, Any]]]]:
        """
        Get all samples with their associated posteriors.
        
        Returns:
            List of (sample, posterior) tuples in chronological order
        """
        samples_with_posteriors = []
        
        # Add observations with posteriors
        for sample, posterior in zip(self._observations, self._obs_posteriors):
            samples_with_posteriors.append((sample, posterior))
        
        # Add intervention outcomes with posteriors
        for (_, outcome), posterior in zip(self._interventions, self._int_posteriors):
            samples_with_posteriors.append((outcome, posterior))
        
        return samples_with_posteriors
    
    def get_posteriors(self) -> List[Optional[Dict[str, Any]]]:
        """Get all posteriors in chronological order."""
        all_posteriors = self._obs_posteriors.copy()
        all_posteriors.extend(self._int_posteriors)
        return all_posteriors
    
    # Filtering operations
    def filter_by_variables(self, variables: FrozenSet[str]) -> 'ExperienceBuffer':
        """
        Create a filtered view of the buffer containing only samples with specified variables.
        
        Args:
            variables: Set of variable names to filter by
            
        Returns:
            New ExperienceBuffer containing only matching samples
        """
        filtered_buffer = ExperienceBuffer()
        
        # Filter observations
        for sample in self._observations:
            sample_vars = frozenset(get_values(sample).keys())
            if variables.issubset(sample_vars):
                filtered_buffer.add_observation(sample)
        
        # Filter interventions
        for intervention, outcome in self._interventions:
            outcome_vars = frozenset(get_values(outcome).keys())
            if variables.issubset(outcome_vars):
                filtered_buffer.add_intervention(intervention, outcome)
        
        logger.debug(f"Filtered buffer: {filtered_buffer.size()} samples with variables {variables}")
        return filtered_buffer
    
    def filter_interventions_by_targets(self, targets: FrozenSet[str]) -> List[Tuple[Intervention, Sample]]:
        """Get intervention-outcome pairs that target specific variables."""
        if targets in self._int_by_targets:
            indices = self._int_by_targets[targets]
            return [self._interventions[i] for i in indices]
        return []
    
    def filter_interventions_by_type(self, intervention_type: str) -> List[Tuple[Intervention, Sample]]:
        """Get intervention-outcome pairs of a specific type."""
        if intervention_type in self._int_by_type:
            indices = self._int_by_type[intervention_type]
            return [self._interventions[i] for i in indices]
        return []
    
    # Batch processing
    def get_observation_batch(self, indices: List[int]) -> List[Sample]:
        """
        Get a batch of observational samples by indices.
        
        Args:
            indices: List of observation indices
            
        Returns:
            List of observational samples
            
        Raises:
            IndexError: If any index is out of range
        """
        if not indices:
            return []
        
        max_idx = max(indices)
        if max_idx >= len(self._observations):
            raise IndexError(f"Observation index {max_idx} out of range (max: {len(self._observations)-1})")
        
        return [self._observations[i] for i in indices]
    
    def get_intervention_batch(self, indices: List[int]) -> List[Tuple[Intervention, Sample]]:
        """
        Get a batch of intervention-outcome pairs by indices.
        
        Args:
            indices: List of intervention indices
            
        Returns:
            List of intervention-outcome pairs
            
        Raises:
            IndexError: If any index is out of range
        """
        if not indices:
            return []
        
        max_idx = max(indices)
        if max_idx >= len(self._interventions):
            raise IndexError(f"Intervention index {max_idx} out of range (max: {len(self._interventions)-1})")
        
        return [self._interventions[i] for i in indices]
    
    def batch_iterator(self, batch_size: int, include_interventions: bool = True) -> Iterator[List[Sample]]:
        """
        Iterate over all samples in batches.
        
        Args:
            batch_size: Size of each batch
            include_interventions: Whether to include intervention outcomes
            
        Yields:
            Batches of samples
        """
        all_samples = self.get_all_samples() if include_interventions else self.get_observations()
        
        for i in range(0, len(all_samples), batch_size):
            yield all_samples[i:i + batch_size]
    
    # Statistics and metadata
    def size(self) -> int:
        """Get total number of samples (observations + interventions)."""
        return len(self._observations) + len(self._interventions)
    
    def num_observations(self) -> int:
        """Get number of observational samples."""
        return len(self._observations)
    
    def num_interventions(self) -> int:
        """Get number of intervention-outcome pairs."""
        return len(self._interventions)
    
    def get_variable_coverage(self) -> FrozenSet[str]:
        """Get set of all variables that appear in any sample."""
        return frozenset(self._variable_coverage)
    
    def get_intervention_types(self) -> FrozenSet[str]:
        """Get set of all intervention types in the buffer."""
        return frozenset(self._int_by_type.keys())
    
    def get_intervention_targets_coverage(self) -> FrozenSet[str]:
        """Get set of all variables that have been intervention targets."""
        all_targets = set()
        for target_set in self._int_by_targets.keys():
            all_targets.update(target_set)
        return frozenset(all_targets)
    
    def get_statistics(self) -> BufferStatistics:
        """Get comprehensive statistics about the buffer."""
        return BufferStatistics(
            total_samples=self.size(),
            num_observations=self.num_observations(),
            num_interventions=self.num_interventions(),
            unique_variables=len(self._variable_coverage),
            unique_intervention_types=len(self.get_intervention_types()),
            unique_intervention_targets=len(self.get_intervention_targets_coverage()),
            creation_time=self._creation_time,
            last_update_time=self._last_update_time
        )
    
    # Validation and debugging
    def validate_consistency(self) -> bool:
        """
        Validate internal consistency of the buffer.
        
        Returns:
            True if buffer is consistent, False otherwise
        """
        try:
            # Check that all stored samples are valid
            for sample in self._observations:
                if not is_observational(sample):
                    logger.error(f"Found non-observational sample in observations: {sample}")
                    return False
            
            for intervention, outcome in self._interventions:
                if not is_interventional(outcome):
                    logger.error(f"Found non-interventional outcome in interventions: {outcome}")
                    return False
            
            # Check index consistency
            total_obs = len(self._observations)
            total_int = len(self._interventions)
            
            # Verify observation indices
            for indices in self._obs_by_variables.values():
                if any(i >= total_obs for i in indices):
                    logger.error("Found out-of-range observation index")
                    return False
            
            # Verify intervention indices
            for indices in self._int_by_targets.values():
                if any(i >= total_int for i in indices):
                    logger.error("Found out-of-range intervention index")
                    return False
            
            for indices in self._int_by_type.values():
                if any(i >= total_int for i in indices):
                    logger.error("Found out-of-range intervention index")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during buffer validation: {e}")
            return False
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a human-readable summary of the buffer contents.
        
        Returns:
            Dictionary with summary information
        """
        stats = self.get_statistics()
        
        return {
            'total_samples': stats.total_samples,
            'observational_samples': stats.num_observations,
            'interventional_samples': stats.num_interventions,
            'unique_variables': stats.unique_variables,
            'variable_coverage': sorted(self._variable_coverage),
            'intervention_types': sorted(self.get_intervention_types()),
            'intervention_targets': sorted(self.get_intervention_targets_coverage()),
            'created_at': time.ctime(stats.creation_time),
            'last_updated': time.ctime(stats.last_update_time),
            'is_consistent': self.validate_consistency()
        }
    
    def __len__(self) -> int:
        """Support len() operation."""
        return self.size()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ExperienceBuffer(observations={self.num_observations()}, "
                f"interventions={self.num_interventions()}, "
                f"variables={len(self._variable_coverage)})")


# Factory functions for common use cases
def create_empty_buffer() -> ExperienceBuffer:
    """Create an empty experience buffer."""
    return ExperienceBuffer()


def create_buffer_from_samples(
    observations: List[Sample],
    interventions: Optional[List[Tuple[Intervention, Sample]]] = None
) -> ExperienceBuffer:
    """
    Create a buffer and populate it with existing samples.
    
    Args:
        observations: List of observational samples
        interventions: Optional list of intervention-outcome pairs
        
    Returns:
        Populated ExperienceBuffer
    """
    buffer = ExperienceBuffer()
    
    # Add observations
    for sample in observations:
        buffer.add_observation(sample)
    
    # Add interventions if provided
    if interventions:
        for intervention, outcome in interventions:
            buffer.add_intervention(intervention, outcome)
    
    return buffer
