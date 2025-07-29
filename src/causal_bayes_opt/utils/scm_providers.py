"""
SCM Provider Utilities

This module provides convenient functions for creating and loading SCMs
for training and evaluation in various formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import pickle

from ..experiments.variable_scm_factory import VariableSCMFactory
from ..experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)

logger = logging.getLogger(__name__)


def create_toy_scm_rotation(
    variable_range: Tuple[int, int] = (3, 5),
    structure_types: List[str] = ["fork", "chain", "collider"],
    samples_per_config: int = 1,
    seed: int = 42
) -> List[Tuple[str, Any]]:
    """
    Create toy SCMs for quick demos and testing.
    
    Args:
        variable_range: Range of variable counts (min, max)
        structure_types: Types of structures to generate
        samples_per_config: Number of SCMs per configuration
        seed: Random seed
        
    Returns:
        List of (name, scm) tuples
    """
    factory = VariableSCMFactory(
        noise_scale=0.5,
        coefficient_range=(-2.0, 2.0),
        seed=seed
    )
    
    scms = []
    
    for num_vars in range(variable_range[0], variable_range[1] + 1):
        for structure_type in structure_types:
            for i in range(samples_per_config):
                try:
                    scm = factory.create_variable_scm(
                        num_variables=num_vars,
                        structure_type=structure_type,
                        target_variable=None  # Auto-select
                    )
                    name = f"{structure_type}_{num_vars}var"
                    if samples_per_config > 1:
                        name += f"_v{i}"
                    scms.append((name, scm))
                except Exception as e:
                    logger.warning(f"Failed to create {structure_type} SCM with {num_vars} vars: {e}")
                    
    logger.info(f"Created {len(scms)} toy SCMs")
    return scms


def load_scm_dataset(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load predefined SCM dataset from disk.
    
    Args:
        path: Path to dataset file or directory
        
    Returns:
        Dictionary mapping names to SCMs
    """
    path = Path(path)
    
    if path.is_file():
        # Load single file
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # Convert list to dict with generated names
            return {f"scm_{i}": scm for i, scm in enumerate(data)}
        else:
            raise ValueError(f"Unsupported data format in {path}")
            
    elif path.is_dir():
        # Load all pickle files from directory
        scms = {}
        
        for file_path in sorted(path.glob("*.pkl")):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    
                base_name = file_path.stem
                
                if isinstance(data, dict):
                    # Add prefix to avoid name collisions
                    for name, scm in data.items():
                        scms[f"{base_name}_{name}"] = scm
                elif isinstance(data, list):
                    for i, scm in enumerate(data):
                        scms[f"{base_name}_{i}"] = scm
                else:
                    scms[base_name] = data
                    
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                
        logger.info(f"Loaded {len(scms)} SCMs from {path}")
        return scms
        
    else:
        raise ValueError(f"Path does not exist: {path}")


def create_benchmark_scms(
    benchmark_name: str = "standard",
    include_variants: bool = True
) -> List[Tuple[str, Any]]:
    """
    Get standard benchmark SCMs.
    
    Args:
        benchmark_name: Name of benchmark set
        include_variants: Whether to include parameter variants
        
    Returns:
        List of (name, scm) tuples
    """
    scms = []
    
    if benchmark_name == "standard":
        # Standard 3-variable benchmarks
        scms.append(("fork_standard", create_fork_scm(noise_scale=1.0, target="Y")))
        scms.append(("chain_standard", create_chain_scm(chain_length=3, coefficient=1.5, noise_scale=1.0)))
        scms.append(("collider_standard", create_collider_scm(noise_scale=1.0)))
        
        if include_variants:
            # Add variants with different parameters
            scms.append(("fork_low_noise", create_fork_scm(noise_scale=0.5, target="Y")))
            scms.append(("fork_high_noise", create_fork_scm(noise_scale=2.0, target="Y")))
            scms.append(("chain_weak", create_chain_scm(chain_length=3, coefficient=0.5, noise_scale=1.0)))
            scms.append(("chain_strong", create_chain_scm(chain_length=3, coefficient=2.5, noise_scale=1.0)))
            
    elif benchmark_name == "extended":
        # Use variable factory for more diverse SCMs
        factory_scms = create_toy_scm_rotation(
            variable_range=(3, 6),
            structure_types=["fork", "chain", "collider", "mixed"],
            samples_per_config=2
        )
        scms.extend(factory_scms)
        
    elif benchmark_name == "minimal":
        # Just the basics for quick testing
        scms.append(("fork_3var", create_fork_scm(noise_scale=1.0, target="Y")))
        scms.append(("chain_3var", create_chain_scm(chain_length=3, coefficient=1.5, noise_scale=1.0)))
        
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
    logger.info(f"Created {len(scms)} benchmark SCMs for '{benchmark_name}'")
    return scms


def save_scm_dataset(
    scms: Union[List[Tuple[str, Any]], Dict[str, Any]], 
    path: Union[str, Path]
) -> None:
    """
    Save SCM dataset to disk.
    
    Args:
        scms: SCMs to save (list of tuples or dict)
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict format
    if isinstance(scms, list):
        scm_dict = {name: scm for name, scm in scms}
    else:
        scm_dict = scms
        
    with open(path, "wb") as f:
        pickle.dump(scm_dict, f)
        
    logger.info(f"Saved {len(scm_dict)} SCMs to {path}")


def create_mixed_dataset(
    n_toy: int = 10,
    n_benchmark: int = 5,
    custom_scms: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Create a mixed dataset with toy, benchmark, and custom SCMs.
    
    Args:
        n_toy: Number of toy SCMs to generate
        n_benchmark: Number of benchmark SCMs to include
        custom_scms: Optional custom SCMs to add
        seed: Random seed
        
    Returns:
        Dictionary mapping names to SCMs
    """
    all_scms = {}
    
    # Add toy SCMs
    if n_toy > 0:
        toy_scms = create_toy_scm_rotation(
            variable_range=(3, 6),
            structure_types=["fork", "chain", "collider", "mixed"],
            samples_per_config=max(1, n_toy // 12),  # Distribute across configs
            seed=seed
        )
        for name, scm in toy_scms[:n_toy]:
            all_scms[f"toy_{name}"] = scm
            
    # Add benchmark SCMs
    if n_benchmark > 0:
        benchmark_scms = create_benchmark_scms("standard", include_variants=True)
        for name, scm in benchmark_scms[:n_benchmark]:
            all_scms[f"benchmark_{name}"] = scm
            
    # Add custom SCMs
    if custom_scms:
        for name, scm in custom_scms.items():
            all_scms[f"custom_{name}"] = scm
            
    logger.info(f"Created mixed dataset with {len(all_scms)} SCMs")
    return all_scms


# Example usage patterns
if __name__ == "__main__":
    # Quick demo
    toy_scms = create_toy_scm_rotation(variable_range=(3, 4))
    print(f"Created {len(toy_scms)} toy SCMs")
    
    # Benchmark set
    benchmark_scms = create_benchmark_scms("standard")
    print(f"Created {len(benchmark_scms)} benchmark SCMs")
    
    # Mixed dataset
    mixed = create_mixed_dataset(n_toy=20, n_benchmark=10)
    print(f"Created mixed dataset with {len(mixed)} SCMs")
    
    # Save and load
    save_scm_dataset(mixed, "data/mixed_scms.pkl")
    loaded = load_scm_dataset("data/mixed_scms.pkl")
    print(f"Loaded {len(loaded)} SCMs")