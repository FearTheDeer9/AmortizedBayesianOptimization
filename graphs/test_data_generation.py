import numpy as np
import networkx as nx
from typing import Dict, List
import pytest

from .graph_generators import (
    generate_erdos_renyi,
    generate_scale_free,
    generate_small_world,
    verify_graph_properties
)
from .scm_generators import (
    generate_linear_scm,
    generate_nonlinear_scm,
    sample_observational,
    sample_interventional
)
from .causal_dataset import CausalDataset


def test_graph_generation():
    """Test basic graph generation properties."""
    # Test Erdos-Renyi
    G = generate_erdos_renyi(10, 0.3, seed=42)
    is_valid, issues = verify_graph_properties(G)
    assert is_valid, f"Erdos-Renyi graph failed validation: {issues}"
    assert len(G.nodes()) == 10

    # Test Scale-Free
    G = generate_scale_free(10, 1.0, seed=42)
    is_valid, issues = verify_graph_properties(G)
    assert is_valid, f"Scale-free graph failed validation: {issues}"
    assert len(G.nodes()) == 10

    # Test Small-World
    G = generate_small_world(10, 2, 0.1, seed=42)
    is_valid, issues = verify_graph_properties(G)
    assert is_valid, f"Small-world graph failed validation: {issues}"
    assert len(G.nodes()) == 10


def test_scm_generation():
    """Test SCM generation and sampling."""
    # Create a simple chain graph
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])

    # Test linear SCM
    scm = generate_linear_scm(G, weight_range=(-1, 1), noise_std=0.1, seed=42)
    samples = sample_observational(scm, 100, seed=42)

    assert len(samples) == 3  # 3 nodes
    assert all(isinstance(v, np.ndarray) for v in samples.values())
    assert all(v.shape == (100, 1) for v in samples.values())

    # Test nonlinear SCM
    scm = generate_nonlinear_scm(
        G, mechanism_type='mlp', hidden_dims=[10], seed=42)
    samples = sample_observational(scm, 100, seed=42)

    assert len(samples) == 3
    assert all(isinstance(v, np.ndarray) for v in samples.values())
    assert all(v.shape == (100, 1) for v in samples.values())


def test_interventions():
    """Test interventional sampling."""
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2)])

    scm = generate_linear_scm(G, seed=42)

    # Test observational sampling
    obs_samples = sample_observational(scm, 100, seed=42)

    # Test interventional sampling
    int_samples = sample_interventional(scm, '1', 5.0, 100, seed=42)

    # Check that intervened node has correct value
    assert np.allclose(int_samples['1'], 5.0)

    # Check that other nodes have different values than observational
    assert not np.allclose(obs_samples['2'], int_samples['2'])


def test_causal_dataset():
    """Test CausalDataset class."""
    G = generate_erdos_renyi(5, 0.3, seed=42)
    scm = generate_linear_scm(G, seed=42)

    dataset = CausalDataset(G, scm, n_obs=200, n_int=50, seed=42)

    # Test observational data
    obs_data = dataset.get_obs_data()
    assert len(obs_data) == 5
    assert all(v.shape == (200, 1) for v in obs_data.values())

    # Test adding interventions
    dataset.add_intervention('0', 1.0)
    dataset.add_intervention('1', 2.0)

    int_data = dataset.get_int_data()
    assert len(int_data) == 2

    # Test getting node-specific data
    node_data, int_node_data = dataset.get_node_data('0')
    assert node_data.shape == (200, 1)
    assert len(int_node_data) == 1
    assert np.allclose(int_node_data[1.0], 1.0)

    # Test getting intervention values
    int_values = dataset.get_intervention_values('0')
    assert len(int_values) == 1
    assert np.allclose(int_values, [1.0])


def test_reproducibility():
    """Test that results are reproducible with the same seed."""
    G = generate_erdos_renyi(5, 0.3, seed=42)
    scm = generate_linear_scm(G, seed=42)

    # Generate two datasets with same seed
    dataset1 = CausalDataset(G, scm, seed=42)
    dataset2 = CausalDataset(G, scm, seed=42)

    # Check observational data matches
    obs1 = dataset1.get_obs_data()
    obs2 = dataset2.get_obs_data()
    for node in obs1:
        assert np.allclose(obs1[node], obs2[node])

    # Check interventions match
    dataset1.add_intervention('0', 1.0)
    dataset2.add_intervention('0', 1.0)

    int1 = dataset1.get_int_data()
    int2 = dataset2.get_int_data()
    for (node, value), data1 in int1.items():
        data2 = int2[(node, value)]
        for n in data1:
            assert np.allclose(data1[n], data2[n])


if __name__ == '__main__':
    pytest.main([__file__])
