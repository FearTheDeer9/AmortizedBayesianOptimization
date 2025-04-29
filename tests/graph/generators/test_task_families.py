#!/usr/bin/env python3
"""
Unit tests for the TaskFamilyGenerator.
"""

from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.graph.generators.task_families import generate_task_family, TaskFamilyGenerationError
from causal_meta.graph.generators.predefined import PredefinedGraphStructureGenerator
import unittest
import random
import os
import sys
import pytest
import numpy as np
import networkx as nx
from causal_meta.utils.graph_utils import calculate_graph_edit_distance

# Add project root to path to allow importing causal_meta
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# --- Fixtures --- 

@pytest.fixture
def base_dag_fixture():
    """Provides a simple base DAG with edge weights."""
    # Use GraphFactory to ensure consistency
    graph = GraphFactory.create_random_dag(num_nodes=6, edge_probability=0.5, seed=42)
    graph.task_id = "BaseDAGFixture"
    # Initialize weights using the new method
    for u, v in graph.get_edges():
        graph.set_edge_attribute(u, v, 'weight', 1.0) # Use the new method
    return graph

# --- Test Cases --- 

def test_generate_family_invalid_inputs(base_dag_fixture):
    """Test generation fails with invalid parameters."""
    base_graph = base_dag_fixture
    with pytest.raises(TaskFamilyGenerationError):
        generate_task_family(base_graph, num_tasks=3, variation_type='invalid_type')
    with pytest.raises(TaskFamilyGenerationError):
        generate_task_family(base_graph, num_tasks=-1)
    with pytest.raises(TaskFamilyGenerationError):
        generate_task_family(base_graph, num_tasks=3, variation_strength=-0.1)
    with pytest.raises(TaskFamilyGenerationError):
        generate_task_family(base_graph, num_tasks=3, variation_strength=1.1)
    with pytest.raises(TypeError):
        generate_task_family("not_a_graph", num_tasks=3)

def test_edge_weight_variation(base_dag_fixture):
    """Test edge weight variation preserves structure and changes weights."""
    base_graph = base_dag_fixture
    num_tasks = 5
    variation_strength = 0.5
    
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='edge_weights', 
                                  variation_strength=variation_strength,
                                  seed=123)
    
    assert len(family) == num_tasks
    base_edges = set(base_graph.get_edges())
    base_nodes = set(base_graph.get_nodes())
    weights_changed_count = 0

    for i, variant in enumerate(family):
        assert isinstance(variant, CausalGraph)
        assert base_nodes == set(variant.get_nodes()), f"Variant {i} nodes differ."
        assert base_edges == set(variant.get_edges()), f"Variant {i} structure differs."
        
        # Check if at least one weight changed compared to base
        one_weight_changed = False
        for u, v in base_edges:
            base_weight = base_graph.get_edge_attribute(u, v, 'weight')
            variant_weight = variant.get_edge_attribute(u, v, 'weight')
            if not np.isclose(base_weight, variant_weight):
                one_weight_changed = True
                break # Found one change, enough for this variant
        
        if one_weight_changed:
            weights_changed_count += 1
            
    # Expect most/all variants to have changed weights if strength > 0
    assert weights_changed_count >= num_tasks * 0.8, "Too few variants had weight changes."

def test_structure_variation(base_dag_fixture):
    """Test structure variation changes edges and preserves DAG property."""
    base_graph = base_dag_fixture
    num_tasks = 5
    variation_strength = 0.3 # Moderate strength
    
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='structure', 
                                  variation_strength=variation_strength,
                                  seed=456)

    assert len(family) == num_tasks
    base_edges = set(base_graph.get_edges())
    base_nodes = set(base_graph.get_nodes())
    structure_changed_count = 0

    for i, variant in enumerate(family):
        assert isinstance(variant, CausalGraph)
        assert base_nodes == set(variant.get_nodes()), f"Variant {i} nodes differ."
        variant_edges = set(variant.get_edges())
        
        if base_edges != variant_edges:
            structure_changed_count += 1
            
        # Verify DAG property using networkx
        nx_variant = nx.DiGraph()
        nx_variant.add_nodes_from(variant.get_nodes())
        nx_variant.add_edges_from(variant.get_edges())
        assert nx.is_directed_acyclic_graph(nx_variant), f"Variant {i} is not a DAG!"

    # Expect most/all variants to have changed structure if strength > 0
    assert structure_changed_count >= num_tasks * 0.8, "Too few variants had structure changes."

def test_structure_variation_zero_strength(base_dag_fixture):
    """Test structure variation with zero strength makes no changes."""
    base_graph = base_dag_fixture
    num_tasks = 3
    
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='structure', 
                                  variation_strength=0.0, # Zero strength
                                  seed=789)

    assert len(family) == num_tasks
    base_edges = set(base_graph.get_edges())

    for variant in family:
        assert base_edges == set(variant.get_edges()), "Structure changed with zero strength!"

def test_node_function_variation_placeholder(base_dag_fixture):
    """Check that node function variation is currently a placeholder."""
    base_graph = base_dag_fixture
    num_tasks = 3
    # Expecting this to run without error but return unmodified graphs (or log warning)
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='node_function', 
                                  seed=101)
    assert len(family) == num_tasks
    # Check structure is unchanged (as function variation shouldn't change it)
    base_edges = set(base_graph.get_edges())
    for variant in family:
         assert base_edges == set(variant.get_edges()), "Node function variation changed structure!"

# Add more tests: different graph sizes, edge cases, higher variation strengths etc.

# --- Integration Tests --- 

@pytest.mark.parametrize("num_nodes, edge_probability, var_strength", [
    (10, 0.2, 0.1), # Small, sparse
    (10, 0.6, 0.4), # Small, dense
    (25, 0.1, 0.2), # Medium, sparse
    (25, 0.4, 0.5), # Medium, dense
])
def test_family_similarity_structure(num_nodes, edge_probability, var_strength):
    """Test that structure variation produces families with measurable similarity.
    Uses graph edit distance (GED).
    """
    base_graph = GraphFactory.create_random_dag(num_nodes=num_nodes, 
                                              edge_probability=edge_probability, 
                                              seed=num_nodes + int(edge_probability*10))
    num_tasks = 4
    
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='structure', 
                                  variation_strength=var_strength,
                                  seed=111)
    
    assert len(family) == num_tasks

    distances = []
    base_nx = base_graph.to_networkx()
    for i, variant in enumerate(family):
        assert isinstance(variant, CausalGraph)
        variant_nx = variant.to_networkx()
        # Ensure DAG property is maintained
        assert nx.is_directed_acyclic_graph(variant_nx), f"Variant {i} is not a DAG!"

        # Calculate similarity to base graph
        # Note: GED can be slow. Consider timeout or sampling for large graphs.
        ged = calculate_graph_edit_distance(base_nx, variant_nx, timeout=30) # Add timeout
        distances.append(ged)
        
    avg_distance = np.mean(distances) if distances else 0
    print(f"Nodes: {num_nodes}, Prob: {edge_probability}, Strength: {var_strength} -> Avg GED: {avg_distance:.2f}")

    # Basic check: average distance should generally increase with variation strength
    # (This is a weak check, needs more runs/stats for reliability)
    # We expect *some* distance if strength > 0
    if var_strength > 0.05: 
        assert avg_distance > 0, "Expected some graph edit distance with non-zero strength"
    else: # Strength is very low, expect very low distance
        assert np.isclose(avg_distance, 0, atol=1e-1), "Expected near-zero distance for low strength"


@pytest.mark.parametrize("num_nodes, edge_probability, var_strength", [
    (10, 0.3, 0.1),
    (10, 0.3, 0.6),
    (20, 0.2, 0.2),
    (20, 0.2, 0.8),
])
def test_family_similarity_weights(num_nodes, edge_probability, var_strength):
    """Test that edge weight variation produces families with weight differences.
    Checks average weight difference magnitude.
    """
    base_graph = GraphFactory.create_random_dag(num_nodes=num_nodes, 
                                              edge_probability=edge_probability, 
                                              seed=num_nodes + int(edge_probability*10))
    # Initialize weights
    for u, v in base_graph.get_edges():
        base_graph.set_edge_attribute(u, v, 'weight', random.uniform(0.5, 1.5))
    num_tasks = 4
    
    family = generate_task_family(base_graph, 
                                  num_tasks=num_tasks, 
                                  variation_type='edge_weights', 
                                  variation_strength=var_strength,
                                  seed=222)
    
    assert len(family) == num_tasks

    avg_weight_diffs = []
    base_edges = list(base_graph.get_edges())

    for i, variant in enumerate(family):
        assert isinstance(variant, CausalGraph)
        # Structure should NOT change for weight variation
        assert set(base_edges) == set(variant.get_edges()), f"Variant {i} structure changed!"
        
        total_diff = 0
        edge_count = 0
        for u, v in base_edges:
            base_weight = base_graph.get_edge_attribute(u, v, 'weight')
            variant_weight = variant.get_edge_attribute(u, v, 'weight')
            total_diff += abs(base_weight - variant_weight)
            edge_count += 1
            
        avg_diff = total_diff / edge_count if edge_count > 0 else 0
        avg_weight_diffs.append(avg_diff)
        
    overall_avg_diff = np.mean(avg_weight_diffs) if avg_weight_diffs else 0
    print(f"Nodes: {num_nodes}, Prob: {edge_probability}, Strength: {var_strength} -> Avg Weight Diff: {overall_avg_diff:.4f}")

    # Expect average weight difference to correlate with variation strength
    if var_strength > 0.05:
        assert overall_avg_diff > 1e-3, "Expected some weight difference with non-zero strength"
    else:
        assert np.isclose(overall_avg_diff, 0, atol=1e-3), "Expected near-zero weight difference for low strength"

# TODO: Add tests for error handling improvements (Item 7)
# TODO: Add tests using predefined graphs (Item 4)
# TODO: Consider property-based testing with Hypothesis (Added Info)
# TODO: Add tests for framework compatibility (Item 9) - may require mocking
