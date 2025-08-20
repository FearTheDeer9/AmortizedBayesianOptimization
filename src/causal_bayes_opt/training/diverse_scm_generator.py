"""
Diverse SCM Generator for random SCM generation during training.

This module provides a simple, efficient generator for creating diverse
SCMs on-the-fly with configurable complexity and structure.
"""

import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np

from ..experiments.variable_scm_factory import VariableSCMFactory

logger = logging.getLogger(__name__)


class DiverseSCMGenerator:
    """
    Generates diverse SCMs on-the-fly for training.
    
    This is the standard generator for random SCM generation, providing:
    - Configurable number of variables (min_vars to max_vars)
    - Various structure types (chain, fork, collider, mixed, random)
    - Varying edge densities based on structure
    - Optional intervention range variation
    """
    
    def __init__(self, 
                 min_vars: int = 3, 
                 max_vars: int = 30, 
                 seed: int = 42,
                 vary_intervention_ranges: bool = True):
        """
        Initialize the generator.
        
        Args:
            min_vars: Minimum number of variables in generated SCMs
            max_vars: Maximum number of variables in generated SCMs
            seed: Random seed for reproducibility
            vary_intervention_ranges: Whether to vary intervention ranges per node/SCM
        """
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.seed = seed
        self.vary_intervention_ranges = vary_intervention_ranges
        
        # Initialize the factory for SCM creation
        self.factory = VariableSCMFactory(
            noise_scale=1.0,
            coefficient_range=(-2.0, 2.0),
            intervention_range=None,  # Let it vary per node/SCM
            vary_intervention_ranges=vary_intervention_ranges,
            seed=seed
        )
        
        # Random state for generation decisions
        self.rng = np.random.RandomState(seed)
        self.generated_count = 0
        
        # Track generation statistics
        self.structure_counts = {
            'collider': 0,
            'fork': 0,
            'chain': 0,
            'mixed': 0,
            'random': 0
        }
        self.size_counts = {
            'small': 0,    # 3-10 vars
            'medium': 0,   # 11-20 vars
            'large': 0,    # 21-30 vars
        }
        
    def __call__(self) -> Dict[str, Any]:
        """
        Generate a single SCM with random properties.
        
        Returns:
            SCM dictionary with structure, variables, and metadata
        """
        # Uniformly sample number of variables
        num_vars = self.rng.randint(self.min_vars, self.max_vars + 1)
        
        # Sample structure type with equal probability
        structures = ['collider', 'fork', 'chain', 'mixed', 'random']
        structure = self.rng.choice(structures)
        
        # Vary edge density based on structure
        if structure == 'chain':
            edge_density = 1.0 / (num_vars - 1) if num_vars > 1 else 0.0
        elif structure in ['fork', 'collider']:
            edge_density = self.rng.uniform(0.2, 0.4)
        elif structure == 'mixed':
            edge_density = self.rng.uniform(0.25, 0.35)
        else:  # random
            edges_per_var = self.rng.uniform(1.0, 3.0)
            edge_density = min(edges_per_var / (num_vars - 1), 0.5)
        
        # Create SCM using factory
        scm = self.factory.create_variable_scm(
            num_variables=num_vars,
            structure_type=structure,
            edge_density=edge_density
        )
        
        # Add metadata
        if num_vars <= 10:
            size_category = 'small'
        elif num_vars <= 20:
            size_category = 'medium'
        else:
            size_category = 'large'
        
        # Update statistics
        self.structure_counts[structure] += 1
        self.size_counts[size_category] += 1
        self.generated_count += 1
        
        # Enrich metadata (handle PMap immutability)
        existing_metadata = scm.get('metadata', {})
        enriched_metadata = dict(existing_metadata)  # Convert to mutable dict
        enriched_metadata.update({
            'size_category': size_category,
            'structure_type': structure,
            'num_variables': num_vars,
            'edge_density': edge_density,
            'generation_idx': self.generated_count
        })
        
        # Update SCM with enriched metadata (create new PMap)
        if hasattr(scm, 'update'):
            # It's a PMap
            scm = scm.update({'metadata': enriched_metadata})
        else:
            # It's a regular dict
            scm['metadata'] = enriched_metadata
        
        # Log generation
        if self.generated_count % 10 == 0:
            logger.info(f"Generated {self.generated_count} SCMs so far")
            logger.debug(f"  Structure distribution: {self.structure_counts}")
            logger.debug(f"  Size distribution: {self.size_counts}")
        
        return scm
    
    @classmethod
    def from_fixed_scms(cls, 
                       scms: Union[List[Any], Dict[str, Any]], 
                       seed: int = 42) -> 'DiverseSCMGenerator':
        """
        Create a generator that rotates through fixed SCMs.
        
        Args:
            scms: List or dict of fixed SCMs to rotate through
            seed: Random seed for rotation order
            
        Returns:
            DiverseSCMGenerator instance that yields from fixed set
        """
        generator = cls(seed=seed)
        
        # Store fixed SCMs
        if isinstance(scms, dict):
            generator.fixed_scms = list(scms.values())
            generator.fixed_scm_names = list(scms.keys())
        else:
            generator.fixed_scms = scms
            generator.fixed_scm_names = [f"scm_{i}" for i in range(len(scms))]
        
        generator.fixed_mode = True
        generator.current_idx = 0
        
        # Override __call__ to return from fixed set
        def fixed_call():
            scm = generator.fixed_scms[generator.current_idx]
            name = generator.fixed_scm_names[generator.current_idx]
            generator.current_idx = (generator.current_idx + 1) % len(generator.fixed_scms)
            
            # Add name to metadata
            if isinstance(scm, dict):
                scm['metadata'] = scm.get('metadata', {})
                scm['metadata']['name'] = name
            
            return scm
        
        generator.__call__ = fixed_call
        return generator
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get generation statistics.
        
        Returns:
            Dictionary with generation counts and distributions
        """
        return {
            'total_generated': self.generated_count,
            'structure_distribution': dict(self.structure_counts),
            'size_distribution': dict(self.size_counts),
            'min_vars': self.min_vars,
            'max_vars': self.max_vars
        }
    
    def reset_statistics(self):
        """Reset generation statistics."""
        self.generated_count = 0
        self.structure_counts = {k: 0 for k in self.structure_counts}
        self.size_counts = {k: 0 for k in self.size_counts}
    
    def __repr__(self) -> str:
        return (f"DiverseSCMGenerator(min_vars={self.min_vars}, "
                f"max_vars={self.max_vars}, generated={self.generated_count})")