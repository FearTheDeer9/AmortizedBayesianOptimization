"""
SCM Manager for Evaluation Pipeline

This module provides utilities for managing Structural Causal Models (SCMs)
in evaluation pipelines, including:
- SCM loading from directory or list
- SCM filtering by complexity/size
- SCM creation from configuration
- SCM caching and lazy loading
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field

import pyrsistent as pyr

from ..data_structures.scm import SCM, create_scm, get_variables
from ..utils.scm_providers import create_toy_scm_rotation, load_scm_dataset
from ..experiments.variable_scm_factory import VariableSCMFactory

logger = logging.getLogger(__name__)


@dataclass
class SCMFilter:
    """Filter criteria for SCMs."""
    min_nodes: Optional[int] = None
    max_nodes: Optional[int] = None
    required_tags: List[str] = field(default_factory=list)
    excluded_tags: List[str] = field(default_factory=list)
    custom_filter: Optional[Callable[[SCM], bool]] = None


@dataclass
class SCMMetadata:
    """Metadata for an SCM."""
    name: str
    node_count: int
    tags: List[str] = field(default_factory=list)
    source: str = "unknown"  # 'file', 'catalog', 'generated'
    file_path: Optional[Path] = None
    
    def matches_filter(self, filter_spec: SCMFilter) -> bool:
        """Check if SCM matches filter criteria."""
        # Check node count
        if filter_spec.min_nodes and self.node_count < filter_spec.min_nodes:
            return False
        if filter_spec.max_nodes and self.node_count > filter_spec.max_nodes:
            return False
            
        # Check required tags
        if filter_spec.required_tags:
            if not all(tag in self.tags for tag in filter_spec.required_tags):
                return False
                
        # Check excluded tags
        if filter_spec.excluded_tags:
            if any(tag in self.tags for tag in filter_spec.excluded_tags):
                return False
                
        return True


class SCMManager:
    """
    Manages SCMs for evaluation pipelines.
    
    Features:
    - Load SCMs from directory or predefined list
    - Filter SCMs by various criteria
    - Cache loaded SCMs for efficiency
    - Support lazy loading for large SCM sets
    """
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialize SCM manager.
        
        Args:
            lazy_load: If True, SCMs are loaded on-demand rather than upfront
        """
        self.lazy_load = lazy_load
        self._scm_cache: Dict[str, SCM] = {}
        self._metadata_cache: Dict[str, SCMMetadata] = {}
        self._scm_sources: Dict[str, Any] = {}  # name -> source info
        
    def add_from_directory(self, directory: Union[str, Path], 
                          pattern: str = "*.pkl",
                          tags: Optional[List[str]] = None) -> int:
        """
        Add SCMs from a directory.
        
        Args:
            directory: Directory containing SCM pickle files
            pattern: File pattern to match
            tags: Tags to assign to these SCMs
            
        Returns:
            Number of SCMs added
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist")
            return 0
            
        files = list(directory.glob(pattern))
        count = 0
        
        for file_path in files:
            try:
                name = file_path.stem
                
                # Store source info for lazy loading
                self._scm_sources[name] = {
                    'type': 'file',
                    'path': file_path
                }
                
                # Load metadata (and optionally the SCM)
                if not self.lazy_load:
                    scm = self._load_scm_from_file(file_path)
                    self._scm_cache[name] = scm
                    metadata = self._extract_metadata(name, scm, tags or [], 'file', file_path)
                else:
                    # For lazy loading, we need to peek at the SCM to get metadata
                    scm = self._load_scm_from_file(file_path)
                    metadata = self._extract_metadata(name, scm, tags or [], 'file', file_path)
                    # Don't cache the SCM itself in lazy mode
                    
                self._metadata_cache[name] = metadata
                count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load SCM from {file_path}: {e}")
                continue
                
        logger.info(f"Added {count} SCMs from {directory}")
        return count
        
    def add_from_toy_rotation(self, 
                             variable_range: Tuple[int, int] = (3, 5),
                             structure_types: List[str] = ["fork", "chain", "collider"],
                             samples_per_config: int = 1,
                             tags: Optional[List[str]] = None,
                             seed: int = 42) -> int:
        """
        Add SCMs from toy rotation generator.
        
        Args:
            variable_range: Range of variable counts (min, max)
            structure_types: Types of structures to generate
            samples_per_config: Number of SCMs per configuration
            tags: Additional tags to assign
            seed: Random seed
            
        Returns:
            Number of SCMs added
        """
        toy_scms = create_toy_scm_rotation(
            variable_range=variable_range,
            structure_types=structure_types,
            samples_per_config=samples_per_config,
            seed=seed
        )
        
        count = 0
        for name, scm in toy_scms:
            try:
                # Store source info
                self._scm_sources[name] = {
                    'type': 'toy',
                    'scm': scm
                }
                
                if not self.lazy_load:
                    self._scm_cache[name] = scm
                    
                # Extract tags from name
                auto_tags = []
                if 'fork' in name:
                    auto_tags.append('fork')
                if 'chain' in name:
                    auto_tags.append('chain')
                if 'collider' in name:
                    auto_tags.append('collider')
                    
                # Add size tags
                n_vars = len(get_variables(scm))
                if n_vars <= 3:
                    auto_tags.append('small')
                elif n_vars >= 5:
                    auto_tags.append('medium')
                    
                all_tags = auto_tags + (tags or [])
                metadata = self._extract_metadata(name, scm, all_tags, 'toy')
                self._metadata_cache[name] = metadata
                count += 1
                
            except Exception as e:
                logger.warning(f"Failed to add toy SCM '{name}': {e}")
                continue
                
        logger.info(f"Added {count} toy SCMs")
        return count
        
    def add_generated(self, configs: List[Dict[str, Any]], 
                     base_name: str = "generated",
                     tags: Optional[List[str]] = None) -> int:
        """
        Add generated SCMs from configurations.
        
        Args:
            configs: List of generation configurations
            base_name: Base name for generated SCMs
            tags: Tags to assign to generated SCMs
            
        Returns:
            Number of SCMs added
        """
        count = 0
        for i, config in enumerate(configs):
            name = f"{base_name}_{i}"
            try:
                # Store source info
                self._scm_sources[name] = {
                    'type': 'generated',
                    'config': config
                }
                
                # Generate SCM using factory
                factory = VariableSCMFactory()
                scm = factory.create_variable_scm(**config)
                if not self.lazy_load:
                    self._scm_cache[name] = scm
                    
                # Auto-tag based on config
                auto_tags = []
                if config.get('n_nodes', 0) <= 5:
                    auto_tags.append('small')
                elif config.get('n_nodes', 0) >= 10:
                    auto_tags.append('large')
                    
                all_tags = auto_tags + (tags or [])
                metadata = self._extract_metadata(name, scm, all_tags, 'generated')
                self._metadata_cache[name] = metadata
                count += 1
                
            except Exception as e:
                logger.warning(f"Failed to generate SCM with config {config}: {e}")
                continue
                
        logger.info(f"Generated {count} SCMs")
        return count
        
    def get_scm(self, name: str) -> Optional[SCM]:
        """
        Get an SCM by name (loading if necessary).
        
        Args:
            name: SCM name
            
        Returns:
            The SCM or None if not found
        """
        # Check cache first
        if name in self._scm_cache:
            return self._scm_cache[name]
            
        # Check if we have source info for lazy loading
        if name not in self._scm_sources:
            logger.warning(f"SCM '{name}' not found")
            return None
            
        # Load the SCM
        source_info = self._scm_sources[name]
        try:
            if source_info['type'] == 'file':
                scm = self._load_scm_from_file(source_info['path'])
            elif source_info['type'] == 'toy':
                scm = source_info['scm']
            elif source_info['type'] == 'generated':
                factory = VariableSCMFactory()
                scm = factory.create_variable_scm(**source_info['config'])
            else:
                raise ValueError(f"Unknown source type: {source_info['type']}")
                
            # Cache for future use
            self._scm_cache[name] = scm
            return scm
            
        except Exception as e:
            logger.error(f"Failed to load SCM '{name}': {e}")
            return None
            
    def list_scms(self, filter_spec: Optional[SCMFilter] = None) -> List[str]:
        """
        List available SCM names, optionally filtered.
        
        Args:
            filter_spec: Optional filter criteria
            
        Returns:
            List of SCM names matching the filter
        """
        names = []
        for name, metadata in self._metadata_cache.items():
            if filter_spec is None or metadata.matches_filter(filter_spec):
                # Apply custom filter if provided
                if filter_spec and filter_spec.custom_filter:
                    scm = self.get_scm(name)
                    if scm and not filter_spec.custom_filter(scm):
                        continue
                names.append(name)
                
        return sorted(names)
        
    def get_metadata(self, name: str) -> Optional[SCMMetadata]:
        """Get metadata for an SCM."""
        return self._metadata_cache.get(name)
        
    def clear_cache(self):
        """Clear the SCM cache (useful for memory management)."""
        self._scm_cache.clear()
        logger.info("Cleared SCM cache")
        
    def _load_scm_from_file(self, path: Path) -> SCM:
        """Load an SCM from a pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def _extract_metadata(self, name: str, scm: SCM, tags: List[str], 
                         source: str, file_path: Optional[Path] = None) -> SCMMetadata:
        """Extract metadata from an SCM."""
        return SCMMetadata(
            name=name,
            node_count=len(get_variables(scm)),
            tags=tags,
            source=source,
            file_path=file_path
        )


def create_default_scm_manager(include_catalog: bool = True,
                              scm_directory: Optional[Union[str, Path]] = None) -> SCMManager:
    """
    Create a default SCM manager with common SCMs.
    
    Args:
        include_catalog: Whether to include SCMs from the built-in catalog
        scm_directory: Optional directory to load additional SCMs from
        
    Returns:
        Configured SCM manager
    """
    manager = SCMManager(lazy_load=True)
    
    # Add toy SCMs for testing
    if include_catalog:
        # Add small SCMs for quick testing
        manager.add_from_toy_rotation(
            variable_range=(3, 3),
            structure_types=['fork', 'chain', 'collider'],
            samples_per_config=1,
            tags=['test', 'small']
        )
        
        # Add medium complexity SCMs
        manager.add_from_toy_rotation(
            variable_range=(4, 5),
            structure_types=['fork', 'chain'],
            samples_per_config=2,
            tags=['benchmark', 'medium']
        )
        
    # Add SCMs from directory if provided
    if scm_directory:
        manager.add_from_directory(scm_directory)
        
    return manager