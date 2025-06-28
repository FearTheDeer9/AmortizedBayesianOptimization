"""
Standard Causal Discovery Dataset Loaders

This module provides functions for loading well-known causal discovery benchmarks
including Sachs protein network, DREAM networks, and BnLearn datasets.
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

# Standard numerical libraries  
import jax.numpy as jnp
import numpy as onp  # For I/O only
import pyrsistent as pyr

# Local imports
from ..data_structures.scm import create_scm
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism
from .test_scms import create_simple_linear_scm

logger = logging.getLogger(__name__)

# Type aliases
DataMatrix = onp.ndarray
GroundTruthGraph = Dict[str, List[str]]

@dataclass(frozen=True)
class BenchmarkDataset:
    """Immutable benchmark dataset specification."""
    name: str
    graph: pyr.PMap  # True causal structure as SCM
    data: Optional[onp.ndarray] = None  # Observational samples [N, d] 
    interventional_data: Optional[Dict[str, onp.ndarray]] = None
    ground_truth_parents: Optional[Dict[str, List[str]]] = None
    target_variable: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _validate_dataset_structure(
    variables: List[str],
    edges: List[Tuple[str, str]],
    data: Optional[onp.ndarray] = None
) -> None:
    """
    Validate dataset structure consistency.
    
    Args:
        variables: List of variable names
        edges: List of (parent, child) edges
        data: Optional data matrix
        
    Raises:
        ValueError: If structure is inconsistent
    """
    if not variables:
        raise ValueError("Variables list cannot be empty")
    
    # Check edge consistency
    variables_set = set(variables)
    for parent, child in edges:
        if parent not in variables_set:
            raise ValueError(f"Edge parent '{parent}' not in variables: {variables}")
        if child not in variables_set:
            raise ValueError(f"Edge child '{child}' not in variables: {variables}")
    
    # Check data consistency
    if data is not None:
        if data.shape[1] != len(variables):
            raise ValueError(f"Data has {data.shape[1]} columns but {len(variables)} variables")
        
        if data.shape[0] == 0:
            raise ValueError("Data matrix cannot be empty")


def _create_synthetic_data(
    scm: pyr.PMap,
    n_samples: int = 1000,
    seed: int = 42
) -> onp.ndarray:
    """
    Generate synthetic data from an SCM for testing purposes.
    
    Args:
        scm: The structural causal model
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        Data matrix [n_samples, n_variables]
    """
    from ..mechanisms.linear import sample_from_linear_scm
    from ..data_structures.scm import get_variables
    from ..data_structures.sample import get_values
    
    # Generate samples
    samples = sample_from_linear_scm(scm, n_samples, seed=seed)
    
    # Convert to matrix format
    variables = sorted(get_variables(scm))
    data = onp.zeros((len(samples), len(variables)))
    
    for i, sample in enumerate(samples):
        values = get_values(sample)
        for j, var in enumerate(variables):
            data[i, j] = values[var]
    
    return data


def create_sachs_dataset(
    use_synthetic: bool = True,
    n_samples: int = 1000,
    noise_scale: float = 0.5,
    seed: int = 42
) -> BenchmarkDataset:
    """
    Create the Sachs protein signaling network dataset.
    
    Based on Sachs et al. (2005) "Causal Protein-Signaling Networks Derived from 
    Multiparameter Single-Cell Data" Science 308:523-529.
    
    The network has 11 proteins with known regulatory relationships.
    
    Args:
        use_synthetic: If True, generate synthetic data based on known structure
        n_samples: Number of samples for synthetic data
        noise_scale: Noise level for synthetic data
        seed: Random seed
        
    Returns:
        BenchmarkDataset with Sachs network structure
        
    Note:
        Real Sachs data requires separate download. This function provides
        the known network structure and can generate synthetic data.
    """
    # Sachs network structure (11 proteins)
    variables = [
        'PKC', 'PKA', 'Raf', 'Mek', 'Erk', 'Akt', 
        'JNK', 'P38', 'PIP2', 'PIP3', 'Plcg'
    ]
    
    # Known causal relationships from the paper
    edges = [
        ('PKC', 'PKA'),     # PKC -> PKA
        ('PKC', 'Raf'),     # PKC -> Raf  
        ('PKA', 'Raf'),     # PKA -> Raf
        ('PKA', 'Mek'),     # PKA -> Mek
        ('PKA', 'Erk'),     # PKA -> Erk
        ('PKA', 'Akt'),     # PKA -> Akt
        ('PKA', 'JNK'),     # PKA -> JNK
        ('PKA', 'P38'),     # PKA -> P38
        ('Raf', 'Mek'),     # Raf -> Mek
        ('Mek', 'Erk'),     # Mek -> Erk
        ('PIP2', 'PIP3'),   # PIP2 -> PIP3
        ('PIP3', 'Akt'),    # PIP3 -> Akt
        ('Plcg', 'PIP2'),   # Plcg -> PIP2
        ('Plcg', 'PKC'),    # Plcg -> PKC
    ]
    
    # Validate structure
    _validate_dataset_structure(variables, edges)
    
    # Create coefficients (based on known positive/negative regulations)
    coefficients = {}
    rng = onp.random.RandomState(seed)
    
    for edge in edges:
        # Most protein regulations are positive in this simplified version
        coeff = rng.uniform(0.5, 2.0)  # Positive regulations
        coefficients[edge] = coeff
    
    # Create noise scales (proteins have different expression variability)
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target='Erk'  # ERK is often the target in pathway analysis
    )
    
    # Add specific metadata
    metadata = {
        'dataset_name': 'sachs',
        'source': 'Sachs et al. (2005) Science',
        'description': 'Protein signaling network (11 proteins)',
        'n_proteins': 11,
        'biological_system': 'T-cell signaling',
        'synthetic_data': use_synthetic,
        'seed': seed if use_synthetic else None
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    # Generate or load data
    data = None
    if use_synthetic:
        data = _create_synthetic_data(scm, n_samples, seed)
        logger.info(f"Generated synthetic Sachs data: {data.shape}")
    else:
        # In real implementation, would load actual Sachs data here
        logger.warning("Real Sachs data not implemented. Use use_synthetic=True")
    
    # Ground truth parent structure
    ground_truth_parents = {}
    for var in variables:
        parents = [parent for parent, child in edges if child == var]
        ground_truth_parents[var] = parents
    
    return BenchmarkDataset(
        name='sachs',
        graph=scm,
        data=data,
        ground_truth_parents=ground_truth_parents,
        target_variable='Erk',
        metadata=metadata
    )


def create_asia_dataset(
    use_synthetic: bool = True,
    n_samples: int = 1000,
    noise_scale: float = 1.0,
    seed: int = 42
) -> BenchmarkDataset:
    """
    Create the Asia Bayesian network dataset.
    
    This is a small medical diagnosis network from the BnLearn repository.
    8 binary variables representing symptoms, conditions, and test results.
    
    Args:
        use_synthetic: If True, generate synthetic continuous data
        n_samples: Number of samples for synthetic data
        noise_scale: Noise level for synthetic data
        seed: Random seed
        
    Returns:
        BenchmarkDataset with Asia network structure
    """
    # Asia network variables (8 nodes)
    variables = ['Asia', 'Smoke', 'Tub', 'Lung', 'Bronc', 'Either', 'Xray', 'Dysp']
    
    # Asia network structure
    edges = [
        ('Asia', 'Tub'),      # Visit to Asia -> Tuberculosis
        ('Smoke', 'Lung'),    # Smoking -> Lung cancer  
        ('Smoke', 'Bronc'),   # Smoking -> Bronchitis
        ('Tub', 'Either'),    # Tuberculosis -> Either (Tub or Lung)
        ('Lung', 'Either'),   # Lung cancer -> Either (Tub or Lung)
        ('Either', 'Xray'),   # Either condition -> X-ray result
        ('Either', 'Dysp'),   # Either condition -> Dyspnea
        ('Bronc', 'Dysp'),    # Bronchitis -> Dyspnea
    ]
    
    # Validate structure
    _validate_dataset_structure(variables, edges)
    
    # Create coefficients  
    coefficients = {}
    base_coeffs = {
        ('Asia', 'Tub'): 1.2,
        ('Smoke', 'Lung'): 1.8,
        ('Smoke', 'Bronc'): 1.5,
        ('Tub', 'Either'): 2.0,
        ('Lung', 'Either'): 2.2,
        ('Either', 'Xray'): 1.7,
        ('Either', 'Dysp'): 1.4,
        ('Bronc', 'Dysp'): 1.3,
    }
    coefficients.update(base_coeffs)
    
    # Create noise scales
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target='Dysp'  # Dyspnea is often the symptom to predict
    )
    
    # Add metadata
    metadata = {
        'dataset_name': 'asia',
        'source': 'BnLearn repository',
        'description': 'Medical diagnosis network (8 variables)',
        'domain': 'medical_diagnosis',
        'original_type': 'discrete',
        'synthetic_continuous': use_synthetic,
        'seed': seed if use_synthetic else None
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    # Generate data
    data = None
    if use_synthetic:
        data = _create_synthetic_data(scm, n_samples, seed)
        logger.info(f"Generated synthetic Asia data: {data.shape}")
    
    # Ground truth parents
    ground_truth_parents = {}
    for var in variables:
        parents = [parent for parent, child in edges if child == var]
        ground_truth_parents[var] = parents
    
    return BenchmarkDataset(
        name='asia',
        graph=scm,
        data=data,
        ground_truth_parents=ground_truth_parents,
        target_variable='Dysp',
        metadata=metadata
    )


def create_alarm_dataset(
    use_synthetic: bool = True,
    n_samples: int = 2000,
    noise_scale: float = 0.8,
    seed: int = 42
) -> BenchmarkDataset:
    """
    Create the ALARM Bayesian network dataset.
    
    Medical monitoring network with 37 variables representing patient monitoring
    in intensive care. This is a larger, more complex network.
    
    Args:
        use_synthetic: If True, generate synthetic continuous data
        n_samples: Number of samples for synthetic data
        noise_scale: Noise level for synthetic data
        seed: Random seed
        
    Returns:
        BenchmarkDataset with ALARM network structure
    """
    # ALARM network variables (37 nodes) - simplified subset for implementation
    # In practice, would load full structure from file
    variables = [
        'HISTORY', 'CVP', 'PCWP', 'HYPOVOLEMIA', 'LVEDVOLUME', 'LVFAILURE',
        'STROKEVOLUME', 'ERRLOWOUTPUT', 'HRBP', 'HREKG', 'ERRCAUTER',
        'HRSAT', 'INSUFFANESTH', 'ANAPHYLAXIS', 'TPR', 'EXPCO2', 'KINKEDTUBE',
        'MINVOL', 'FIO2', 'PVSAT', 'SAO2', 'SHUNT', 'PULMEMBOLUS',
        'PAP', 'PRESS', 'DISCONNECT', 'MINVOLSET', 'VENTMACH', 'VENTTUBE',
        'VENTLUNG', 'VENTALV', 'ARTCO2', 'CATECHOL', 'HR', 'CO', 'BP'
    ]
    
    # Simplified ALARM structure (subset of actual network)
    edges = [
        ('HISTORY', 'LVFAILURE'),
        ('CVP', 'LVEDVOLUME'),
        ('PCWP', 'LVEDVOLUME'),
        ('HYPOVOLEMIA', 'LVEDVOLUME'),
        ('LVEDVOLUME', 'LVFAILURE'),
        ('LVEDVOLUME', 'STROKEVOLUME'),
        ('LVFAILURE', 'STROKEVOLUME'),
        ('STROKEVOLUME', 'CO'),
        ('HR', 'CO'),
        ('CO', 'BP'),
        ('TPR', 'BP'),
        ('INSUFFANESTH', 'TPR'),
        ('ANAPHYLAXIS', 'TPR'),
        ('FIO2', 'PVSAT'),
        ('SHUNT', 'PVSAT'),
        ('PVSAT', 'SAO2'),
        ('PULMEMBOLUS', 'PAP'),
        ('KINKEDTUBE', 'PRESS'),
        ('DISCONNECT', 'PRESS'),
        ('MINVOLSET', 'MINVOL'),
        ('VENTMACH', 'MINVOL'),
        ('VENTTUBE', 'MINVOL'),
        ('MINVOL', 'EXPCO2'),
        ('ARTCO2', 'EXPCO2'),
    ]
    
    # Take only variables that appear in edges for consistency
    edge_vars = set()
    for parent, child in edges:
        edge_vars.add(parent)
        edge_vars.add(child)
    
    variables = sorted(list(edge_vars))
    
    # Validate structure
    _validate_dataset_structure(variables, edges)
    
    # Create coefficients
    coefficients = {}
    rng = onp.random.RandomState(seed)
    
    for edge in edges:
        # Medical parameters have varied effect sizes
        coeff = rng.uniform(0.3, 2.5)
        coefficients[edge] = coeff
    
    # Create noise scales
    noise_scales = {var: noise_scale for var in variables}
    
    # Create SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target='BP'  # Blood pressure is often a key monitoring target
    )
    
    # Add metadata
    metadata = {
        'dataset_name': 'alarm',
        'source': 'BnLearn repository',
        'description': 'Medical monitoring network (subset of 37 variables)',
        'domain': 'medical_monitoring',
        'original_type': 'discrete',
        'synthetic_continuous': use_synthetic,
        'subset_size': len(variables),
        'seed': seed if use_synthetic else None
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    # Generate data
    data = None
    if use_synthetic:
        data = _create_synthetic_data(scm, n_samples, seed)
        logger.info(f"Generated synthetic ALARM data: {data.shape}")
    
    # Ground truth parents
    ground_truth_parents = {}
    for var in variables:
        parents = [parent for parent, child in edges if child == var]
        ground_truth_parents[var] = parents
    
    return BenchmarkDataset(
        name='alarm',
        graph=scm,
        data=data,
        ground_truth_parents=ground_truth_parents,
        target_variable='BP',
        metadata=metadata
    )


def create_dream_network(
    network_size: int = 10,
    network_id: str = "dream4_010",
    use_synthetic: bool = True,
    n_samples: int = 1000,
    noise_scale: float = 0.7,
    seed: int = 42
) -> BenchmarkDataset:
    """
    Create a DREAM challenge gene regulatory network.
    
    DREAM (Dialogue for Reverse Engineering Assessments and Methods) networks
    are synthetic gene regulatory networks used for benchmarking.
    
    Args:
        network_size: Size of the network (10, 50, or 100)
        network_id: Specific DREAM network identifier
        use_synthetic: If True, generate synthetic data
        n_samples: Number of samples for synthetic data
        noise_scale: Noise level for synthetic data  
        seed: Random seed
        
    Returns:
        BenchmarkDataset with DREAM network structure
    """
    if network_size not in [10, 50, 100]:
        raise ValueError(f"network_size must be 10, 50, or 100, got: {network_size}")
    
    # Create variables (genes)
    variables = [f"G{i}" for i in range(1, network_size + 1)]
    
    # Generate synthetic gene regulatory network structure
    # In practice, would load actual DREAM network topologies
    rng = onp.random.RandomState(seed)
    
    # Create approximately 2-3 edges per node on average
    n_edges = min(network_size * 2, network_size * (network_size - 1) // 4)
    
    # Generate edges ensuring DAG property
    edges = []
    for i in range(network_size):
        for j in range(i + 1, network_size):
            if rng.random() < (n_edges / (network_size * (network_size - 1) / 2)):
                edges.append((variables[i], variables[j]))
    
    # Limit to target number of edges
    if len(edges) > n_edges:
        selected_indices = rng.choice(len(edges), size=n_edges, replace=False)
        edges = [edges[i] for i in selected_indices]
    
    # Validate structure
    _validate_dataset_structure(variables, edges)
    
    # Create coefficients (gene regulatory interactions can be positive or negative)
    coefficients = {}
    for edge in edges:
        # 70% positive regulation, 30% negative
        if rng.random() < 0.7:
            coeff = rng.uniform(0.5, 2.0)  # Positive regulation
        else:
            coeff = rng.uniform(-2.0, -0.5)  # Negative regulation
        coefficients[edge] = coeff
    
    # Create noise scales
    noise_scales = {var: noise_scale for var in variables}
    
    # Select target gene (often one with high connectivity)
    gene_connectivity = {}
    for var in variables:
        connectivity = sum(1 for p, c in edges if p == var or c == var)
        gene_connectivity[var] = connectivity
    
    target_variable = max(gene_connectivity, key=gene_connectivity.get)
    
    # Create SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target_variable
    )
    
    # Add metadata
    metadata = {
        'dataset_name': f'dream_{network_size}',
        'network_id': network_id,
        'source': 'DREAM Challenge',
        'description': f'Gene regulatory network ({network_size} genes)',
        'domain': 'gene_regulation',
        'network_size': network_size,
        'n_edges': len(edges),
        'synthetic_structure': True,  # Our implementation uses synthetic structure
        'seed': seed
    }
    
    scm = scm.set('metadata', pyr.pmap(scm['metadata'].update(metadata)))
    
    # Generate data
    data = None
    if use_synthetic:
        data = _create_synthetic_data(scm, n_samples, seed)
        logger.info(f"Generated synthetic DREAM data: {data.shape}")
    
    # Ground truth parents
    ground_truth_parents = {}
    for var in variables:
        parents = [parent for parent, child in edges if child == var]
        ground_truth_parents[var] = parents
    
    return BenchmarkDataset(
        name=f'dream_{network_size}',
        graph=scm,
        data=data,
        ground_truth_parents=ground_truth_parents,
        target_variable=target_variable,
        metadata=metadata
    )


def get_available_datasets() -> List[str]:
    """
    Get list of available benchmark datasets.
    
    Returns:
        List of dataset names that can be loaded
    """
    return ['sachs', 'asia', 'alarm', 'dream_10', 'dream_50', 'dream_100']


def load_benchmark_dataset(
    dataset_name: str,
    use_synthetic: bool = True,
    n_samples: int = 1000,
    **kwargs
) -> BenchmarkDataset:
    """
    Load a benchmark dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        use_synthetic: Whether to use synthetic data generation
        n_samples: Number of samples for synthetic data
        **kwargs: Additional arguments passed to specific dataset loaders
        
    Returns:
        BenchmarkDataset object
        
    Raises:
        ValueError: If dataset_name is not recognized
        
    Example:
        >>> dataset = load_benchmark_dataset('sachs', n_samples=500)
        >>> print(f"Loaded {dataset.name} with {len(dataset.data)} samples")
    """
    dataset_loaders = {
        'sachs': create_sachs_dataset,
        'asia': create_asia_dataset,
        'alarm': create_alarm_dataset,
        'dream_10': lambda **kw: create_dream_network(network_size=10, **kw),
        'dream_50': lambda **kw: create_dream_network(network_size=50, **kw),
        'dream_100': lambda **kw: create_dream_network(network_size=100, **kw),
    }
    
    if dataset_name not in dataset_loaders:
        available = get_available_datasets()
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    loader = dataset_loaders[dataset_name]
    return loader(use_synthetic=use_synthetic, n_samples=n_samples, **kwargs)


def get_dataset_summary(dataset: BenchmarkDataset) -> Dict[str, Any]:
    """
    Get a summary of a benchmark dataset.
    
    Args:
        dataset: The benchmark dataset to summarize
        
    Returns:
        Dictionary containing dataset statistics and properties
    """
    from ..data_structures.scm import get_variables, get_edges
    
    variables = get_variables(dataset.graph)
    edges = get_edges(dataset.graph)
    
    summary = {
        'name': dataset.name,
        'n_variables': len(variables),
        'n_edges': len(edges),
        'target_variable': dataset.target_variable,
        'has_data': dataset.data is not None,
        'data_samples': dataset.data.shape[0] if dataset.data is not None else 0,
        'edge_density': len(edges) / (len(variables) * (len(variables) - 1) / 2) if len(variables) > 1 else 0,
        'metadata': dataset.metadata
    }
    
    if dataset.ground_truth_parents:
        # Compute parent statistics
        parent_counts = [len(parents) for parents in dataset.ground_truth_parents.values()]
        summary.update({
            'max_parents': max(parent_counts) if parent_counts else 0,
            'avg_parents': sum(parent_counts) / len(parent_counts) if parent_counts else 0,
            'root_variables': [var for var, parents in dataset.ground_truth_parents.items() if len(parents) == 0]
        })
    
    return summary


# Export public functions
__all__ = [
    'BenchmarkDataset',
    'create_sachs_dataset',
    'create_asia_dataset', 
    'create_alarm_dataset',
    'create_dream_network',
    'load_benchmark_dataset',
    'get_available_datasets',
    'get_dataset_summary'
]