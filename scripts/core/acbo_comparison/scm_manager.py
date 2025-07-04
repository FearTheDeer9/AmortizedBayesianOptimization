"""
SCM Manager for ACBO Comparison Framework

This module manages SCM generation, caching, and metadata tracking.
It provides a clean interface for generating test SCMs with variable sizes.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
import pickle
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pyrsistent as pyr
from omegaconf import DictConfig

from causal_bayes_opt.experiments.variable_scm_factory import (
    VariableSCMFactory, get_scm_info
)
from causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from causal_bayes_opt.data_structures.scm import get_variables, get_target

logger = logging.getLogger(__name__)


class SCMManager:
    """Manager for SCM generation, caching, and metadata tracking."""
    
    def __init__(self, config: DictConfig, cache_dir: Optional[Path] = None):
        """
        Initialize SCM manager.
        
        Args:
            config: Experiment configuration
            cache_dir: Directory for caching generated SCMs
        """
        self.config = config
        self.cache_dir = cache_dir or Path("scm_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.scms: Dict[str, pyr.PMap] = {}
        self.scm_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize SCM factory
        self.factory = VariableSCMFactory(
            noise_scale=1.0,
            coefficient_range=(-2.0, 2.0),
            seed=getattr(config, 'seed', 42)
        )
        
        logger.info(f"Initialized SCM manager with cache dir: {self.cache_dir}")
    
    def generate_test_scms(self) -> List[Tuple[str, pyr.PMap]]:
        """Generate test SCMs based on configuration."""
        try:
            # Check if using predefined SCM suite
            if self._use_predefined_suite():
                return self._generate_predefined_suite()
            
            # Check if using variable factory
            elif self._use_variable_factory():
                return self._generate_variable_scms()
            
            # Fallback to random generation
            else:
                return self._generate_random_scms()
                
        except Exception as e:
            logger.error(f"Failed to generate test SCMs: {e}")
            # Final fallback to hardcoded SCMs
            return self._generate_fallback_scms()
    
    def get_scm_metadata(self, scm_name: str) -> Dict[str, Any]:
        """Get metadata for a specific SCM."""
        if scm_name not in self.scm_metadata:
            if scm_name in self.scms:
                self.scm_metadata[scm_name] = get_scm_info(self.scms[scm_name])
            else:
                raise ValueError(f"Unknown SCM: {scm_name}")
        
        return self.scm_metadata[scm_name]
    
    def cache_scms(self, scms: List[Tuple[str, pyr.PMap]]) -> None:
        """Cache generated SCMs for reuse."""
        cache_file = self.cache_dir / "scm_cache.pkl"
        
        try:
            # Create cache data
            cache_data = {
                'scms': {name: scm for name, scm in scms},
                'metadata': {name: get_scm_info(scm) for name, scm in scms},
                'config_hash': self._get_config_hash()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cached {len(scms)} SCMs to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache SCMs: {e}")
    
    def load_cached_scms(self) -> Optional[List[Tuple[str, pyr.PMap]]]:
        """Load cached SCMs if available and valid."""
        cache_file = self.cache_dir / "scm_cache.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is valid for current config
            if cache_data.get('config_hash') != self._get_config_hash():
                logger.info("Cache invalid due to config change")
                return None
            
            # Load cached data
            scms = list(cache_data['scms'].items())
            self.scms.update(cache_data['scms'])
            self.scm_metadata.update(cache_data['metadata'])
            
            logger.info(f"Loaded {len(scms)} SCMs from cache")
            return scms
            
        except Exception as e:
            logger.warning(f"Failed to load cached SCMs: {e}")
            return None
    
    def _use_predefined_suite(self) -> bool:
        """Check if configuration specifies predefined SCM suite."""
        return (hasattr(self.config.experiment, 'scm_suite') and
                getattr(self.config.experiment.scm_suite, 'enabled', False))
    
    def _use_variable_factory(self) -> bool:
        """Check if configuration specifies variable factory."""
        return (hasattr(self.config.experiment, 'scm_generation') and
                getattr(self.config.experiment.scm_generation, 'use_variable_factory', True))
    
    def _generate_predefined_suite(self) -> List[Tuple[str, pyr.PMap]]:
        """Generate SCMs from predefined suite."""
        from causal_bayes_opt.experiments.benchmark_scms import create_scm_suite
        
        scm_suite = create_scm_suite()
        selected_scms = getattr(self.config.experiment.scm_suite, 'scm_names', 
                              list(scm_suite.keys())[:3])
        
        scms = [(name, scm_suite[name]) for name in selected_scms if name in scm_suite]
        
        # Store for metadata access
        for name, scm in scms:
            self.scms[name] = scm
        
        logger.info(f"Generated {len(scms)} SCMs from predefined suite")
        return scms
    
    def _generate_variable_scms(self) -> List[Tuple[str, pyr.PMap]]:
        """Generate SCMs using variable factory."""
        scm_config = self.config.experiment.scm_generation
        
        variable_range = getattr(scm_config, 'variable_range', [3, 6])
        structure_types = getattr(scm_config, 'structure_types', 
                                ['fork', 'chain', 'collider', 'mixed'])
        
        scms = []
        
        # Generate SCMs for each combination
        for structure_type in structure_types:
            for num_vars in variable_range:
                scm_name = f"{structure_type}_{num_vars}var"
                
                try:
                    scm = self.factory.create_variable_scm(
                        num_variables=num_vars,
                        structure_type=structure_type
                    )
                    scms.append((scm_name, scm))
                    self.scms[scm_name] = scm
                    
                    logger.info(f"Generated {scm_name}: {get_scm_info(scm)}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {scm_name}: {e}")
        
        if not scms:
            raise ValueError("No variable SCMs could be generated")
        
        logger.info(f"Generated {len(scms)} variable SCMs")
        return scms
    
    def _generate_random_scms(self) -> List[Tuple[str, pyr.PMap]]:
        """Generate random SCMs based on configuration."""
        from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm
        
        n_variables = getattr(self.config.experiment.environment, 'num_variables', 5)
        edge_density = getattr(self.config.experiment.problem, 'edge_density', 0.5)
        n_scms = getattr(self.config, 'n_scms', 2)
        
        scms = []
        for i in range(n_scms):
            scm_name = f"erdos_renyi_{i}"
            scm = create_erdos_renyi_scm(
                n_nodes=n_variables,
                edge_prob=edge_density,
                noise_scale=getattr(self.config.experiment.environment, 'noise_scale', 1.0),
                seed=getattr(self.config, 'seed', 42) + i
            )
            scms.append((scm_name, scm))
            self.scms[scm_name] = scm
        
        logger.info(f"Generated {len(scms)} random SCMs")
        return scms
    
    def _generate_fallback_scms(self) -> List[Tuple[str, pyr.PMap]]:
        """Generate fallback SCMs using hardcoded approach."""
        scms = []
        
        try:
            # Create basic 3-variable SCMs
            scm = create_fork_scm(noise_scale=1.0, target="Y")
            scms.append(("fork_3var", scm))
            self.scms["fork_3var"] = scm
            
            scm = create_chain_scm(chain_length=3, coefficient=1.5, noise_scale=1.0)
            scms.append(("chain_3var", scm))
            self.scms["chain_3var"] = scm
            
            scm = create_fork_scm(noise_scale=1.0, target="Y")  # Collider variant
            scms.append(("collider_3var", scm))
            self.scms["collider_3var"] = scm
            
            logger.info(f"Generated {len(scms)} fallback SCMs")
            
        except Exception as e:
            logger.error(f"Failed to generate fallback SCMs: {e}")
            raise ValueError("Could not generate any SCMs")
        
        return scms
    
    def _get_config_hash(self) -> str:
        """Get hash of relevant configuration for cache validation."""
        # Extract relevant config parts
        config_str = ""
        
        if hasattr(self.config.experiment, 'scm_generation'):
            scm_config = self.config.experiment.scm_generation
            config_str += str(getattr(scm_config, 'variable_range', []))
            config_str += str(getattr(scm_config, 'structure_types', []))
            config_str += str(getattr(scm_config, 'use_variable_factory', True))
        
        if hasattr(self.config.experiment, 'environment'):
            env_config = self.config.experiment.environment
            config_str += str(getattr(env_config, 'num_variables', 5))
            config_str += str(getattr(env_config, 'noise_scale', 1.0))
        
        config_str += str(getattr(self.config, 'seed', 42))
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_scm_characteristics_summary(self) -> Dict[str, Any]:
        """Get summary of all SCM characteristics."""
        if not self.scms:
            return {}
        
        summary = {
            'total_scms': len(self.scms),
            'variable_counts': {},
            'structure_types': {},
            'edge_densities': [],
            'scm_details': {}
        }
        
        for name, scm in self.scms.items():
            info = get_scm_info(scm)
            
            # Count variable distributions
            num_vars = info['num_variables']
            summary['variable_counts'][num_vars] = summary['variable_counts'].get(num_vars, 0) + 1
            
            # Count structure types
            structure_type = info['structure_type']
            summary['structure_types'][structure_type] = summary['structure_types'].get(structure_type, 0) + 1
            
            # Collect edge densities
            summary['edge_densities'].append(info['edge_density'])
            
            # Store detailed info
            summary['scm_details'][name] = info
        
        return summary