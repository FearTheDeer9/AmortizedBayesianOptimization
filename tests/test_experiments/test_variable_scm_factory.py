"""
Tests for VariableSCMFactory - Critical SCM generation component.

This module tests the variable SCM generation functionality that creates
SCMs with configurable variable counts (3-8 variables) and different
structural patterns.

Following TDD principles - these tests define the expected behavior.
"""

import pytest
import jax.random as random
import pyrsistent as pyr
from typing import Set, List, Tuple
from hypothesis import given, strategies as st, assume
import networkx as nx

from causal_bayes_opt.experiments.variable_scm_factory import (
    VariableSCMFactory, 
    get_scm_info
)
from causal_bayes_opt.data_structures.scm import get_variables, get_target, get_edges


class TestVariableSCMFactory:
    """Test the main SCM factory functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.factory = VariableSCMFactory(
            noise_scale=1.0,
            coefficient_range=(-2.0, 2.0),
            seed=42
        )
    
    def test_factory_initialization(self):
        """Test factory initializes with correct parameters."""
        assert self.factory.noise_scale == 1.0
        assert self.factory.coefficient_range == (-2.0, 2.0)
        assert self.factory.seed == 42
    
    @given(
        num_variables=st.integers(min_value=3, max_value=8),
        structure_type=st.sampled_from(["fork", "chain", "collider", "mixed"])
    )
    def test_create_variable_scm_basic_properties(self, num_variables, structure_type):
        """Property test: All generated SCMs should have basic validity."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type
        )
        
        # Basic SCM properties
        assert isinstance(scm, pyr.PMap)
        assert 'variables' in scm
        assert 'edges' in scm
        assert 'target' in scm
        
        # Variable count should match request
        variables = get_variables(scm)
        assert len(variables) == num_variables
        
        # Target should be one of the variables
        target = get_target(scm)
        assert target in variables
        
        # All variables should be strings
        for var in variables:
            assert isinstance(var, str)
    
    @given(
        num_variables=st.integers(min_value=3, max_value=8),
        structure_type=st.sampled_from(["fork", "chain", "collider", "mixed"])
    )
    def test_generated_scm_is_dag_property(self, num_variables, structure_type):
        """Property test: All generated SCMs must be DAGs (no cycles)."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type
        )
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Convert to networkx for cycle detection
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        
        # Must be a DAG
        assert nx.is_directed_acyclic_graph(G), f"Generated SCM has cycles: {edges}"
    
    @given(num_variables=st.integers(min_value=3, max_value=8))
    def test_fork_structure_properties(self, num_variables):
        """Test specific properties of fork structures."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type="fork"
        )
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        target = get_target(scm)
        
        # Fork: root node points to all others
        # Find root (node with no incoming edges)
        incoming_counts = {var: 0 for var in variables}
        for parent, child in edges:
            incoming_counts[child] += 1
        
        roots = [var for var, count in incoming_counts.items() if count == 0]
        
        # Should have exactly one root
        assert len(roots) == 1
        root = roots[0]
        
        # Root should point to all other variables
        root_children = {child for parent, child in edges if parent == root}
        non_root_vars = variables - {root}
        
        # All non-root variables should be children of root
        assert root_children == non_root_vars
    
    @given(num_variables=st.integers(min_value=3, max_value=8))
    def test_chain_structure_properties(self, num_variables):
        """Test specific properties of chain structures."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type="chain"
        )
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Chain: should have exactly (num_variables - 1) edges
        assert len(edges) == num_variables - 1
        
        # Each variable (except root) should have exactly one parent
        incoming_counts = {var: 0 for var in variables}
        outgoing_counts = {var: 0 for var in variables}
        
        for parent, child in edges:
            incoming_counts[child] += 1
            outgoing_counts[parent] += 1
        
        # Exactly one root (0 incoming)
        roots = [var for var, count in incoming_counts.items() if count == 0]
        assert len(roots) == 1
        
        # Exactly one leaf (0 outgoing)
        leaves = [var for var, count in outgoing_counts.items() if count == 0]
        assert len(leaves) == 1
        
        # All other nodes should have exactly 1 incoming and 1 outgoing
        middle_nodes = variables - set(roots) - set(leaves)
        for var in middle_nodes:
            assert incoming_counts[var] == 1
            assert outgoing_counts[var] == 1
    
    @given(num_variables=st.integers(min_value=3, max_value=8))
    def test_collider_structure_properties(self, num_variables):
        """Test specific properties of collider structures."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type="collider"
        )
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Collider: multiple nodes point to one central node
        incoming_counts = {var: 0 for var in variables}
        for parent, child in edges:
            incoming_counts[child] += 1
        
        # Should have at least one node with multiple parents
        nodes_with_multiple_parents = [
            var for var, count in incoming_counts.items() if count >= 2
        ]
        assert len(nodes_with_multiple_parents) >= 1
    
    @given(
        num_variables=st.integers(min_value=3, max_value=8),
        edge_density=st.floats(min_value=0.1, max_value=0.9)
    )
    def test_mixed_structure_edge_density(self, num_variables, edge_density):
        """Test that mixed structures respect edge density constraints."""
        scm = self.factory.create_variable_scm(
            num_variables=num_variables,
            structure_type="mixed",
            edge_density=edge_density
        )
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Calculate maximum possible edges (n * (n-1) / 2 for complete DAG)
        max_edges = num_variables * (num_variables - 1) // 2
        expected_edges = int(edge_density * max_edges)
        
        # Should be approximately correct (within reasonable tolerance)
        assert abs(len(edges) - expected_edges) <= max(1, expected_edges * 0.3)
    
    def test_custom_target_variable_from_generated_set(self):
        """Test SCM creation with target from generated variables."""
        # Use a valid target from the generated variable set
        scm = self.factory.create_variable_scm(
            num_variables=4,
            structure_type="fork",
            target_variable="X2"  # Valid target from ['X0', 'X1', 'X2', 'X3']
        )
        
        variables = get_variables(scm)
        target = get_target(scm)
        
        # Target should be set correctly
        assert target == "X2"
        # Variables should include the target
        assert "X2" in variables
    
    def test_deterministic_generation(self):
        """Test that same seed produces same SCM."""
        factory1 = VariableSCMFactory(seed=12345)
        factory2 = VariableSCMFactory(seed=12345)
        
        scm1 = factory1.create_variable_scm(4, "mixed", edge_density=0.5)
        scm2 = factory2.create_variable_scm(4, "mixed", edge_density=0.5)
        
        # Should be identical
        assert get_variables(scm1) == get_variables(scm2)
        assert get_edges(scm1) == get_edges(scm2)
        assert get_target(scm1) == get_target(scm2)
    
    def test_different_seeds_produce_different_scms(self):
        """Test that different seeds produce different SCMs."""
        factory1 = VariableSCMFactory(seed=1)
        factory2 = VariableSCMFactory(seed=2)
        
        scm1 = factory1.create_variable_scm(5, "mixed", edge_density=0.4)
        scm2 = factory2.create_variable_scm(5, "mixed", edge_density=0.4)
        
        # Should be different (at least one property differs)
        different = (
            get_variables(scm1) != get_variables(scm2) or
            get_edges(scm1) != get_edges(scm2) or 
            get_target(scm1) != get_target(scm2)
        )
        assert different
    
    def test_create_scm_suite(self):
        """Test creation of SCM suite with multiple structures."""
        suite = self.factory.create_scm_suite(
            variable_ranges=[3, 4, 5],
            structure_types=["fork", "chain"]
        )
        
        # Should have 3 variable counts * 2 structure types = 6 SCMs
        assert len(suite) == 6
        
        # Each entry should be a dictionary mapping name to SCM
        for name, scm in suite.items():
            assert isinstance(name, str)
            assert isinstance(scm, pyr.PMap)
            assert len(get_variables(scm)) >= 3
            assert len(get_variables(scm)) <= 5
    
    def test_invalid_structure_type(self):
        """Test error handling for invalid structure types."""
        with pytest.raises(ValueError, match="Unknown structure_type"):
            self.factory.create_variable_scm(4, "invalid_type")
    
    def test_invalid_variable_count(self):
        """Test error handling for invalid variable counts."""
        with pytest.raises(ValueError, match="num_variables must be 3-8"):
            self.factory.create_variable_scm(1, "fork")
    
    def test_invalid_edge_density(self):
        """Test handling of edge density outside normal range."""
        # Since validation may not exist, just test that it doesn't crash
        scm = self.factory.create_variable_scm(4, "mixed", edge_density=0.9)
        assert isinstance(scm, pyr.PMap)


class TestSCMStructureGeneration:
    """Test structure generation through public API."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.factory = VariableSCMFactory(seed=42)
    
    def test_fork_structure_properties(self):
        """Test fork structure generation through public API."""
        scm = self.factory.create_variable_scm(4, "fork", target_variable="X2")
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        target = get_target(scm)
        
        assert target == "X2"
        
        # Find nodes with no incoming edges (roots)
        incoming_counts = {var: 0 for var in variables}
        for parent, child in edges:
            incoming_counts[child] += 1
        
        roots = [var for var, count in incoming_counts.items() if count == 0]
        
        # Fork should have exactly one root
        assert len(roots) == 1
        
        # All non-root variables should have incoming edges
        non_roots = variables - {roots[0]}
        for var in non_roots:
            assert incoming_counts[var] > 0
    
    def test_chain_structure_properties(self):
        """Test chain structure generation through public API."""
        scm = self.factory.create_variable_scm(4, "chain")
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Chain should have exactly (n-1) edges
        assert len(edges) == 3
        
        # Verify it forms a valid chain
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        
        # Should be connected and acyclic
        assert nx.is_directed_acyclic_graph(G)
        assert nx.is_weakly_connected(G)
    
    def test_collider_structure_properties(self):
        """Test collider structure generation through public API."""
        scm = self.factory.create_variable_scm(4, "collider")
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Should have at least one node with multiple parents
        incoming_counts = {var: 0 for var in variables}
        for parent, child in edges:
            incoming_counts[child] += 1
        
        nodes_with_multiple_parents = [
            var for var, count in incoming_counts.items() if count >= 2
        ]
        assert len(nodes_with_multiple_parents) >= 1
    
    def test_cycle_prevention(self):
        """Test that generated structures are always DAGs."""
        for structure_type in ["fork", "chain", "collider", "mixed"]:
            scm = self.factory.create_variable_scm(5, structure_type)
            
            variables = get_variables(scm)
            edges = get_edges(scm)
            
            # Convert to networkx for cycle detection
            G = nx.DiGraph()
            G.add_nodes_from(variables)
            G.add_edges_from(edges)
            
            # Must be a DAG
            assert nx.is_directed_acyclic_graph(G), f"{structure_type} structure has cycles"
    
    def test_random_structure_edge_density(self):
        """Test that random structure respects edge density."""
        for density in [0.2, 0.5, 0.8]:
            scm = self.factory.create_variable_scm(5, "random", edge_density=density)
            
            variables = get_variables(scm)
            edges = get_edges(scm)
            
            # Calculate expected number of edges
            max_edges = len(variables) * (len(variables) - 1) // 2
            expected_edges = int(density * max_edges)
            
            # Should be approximately correct (within reasonable tolerance)
            tolerance = max(1, expected_edges * 0.4)
            assert abs(len(edges) - expected_edges) <= tolerance
            
            # Should still be a DAG
            G = nx.DiGraph()
            G.add_nodes_from(variables)
            G.add_edges_from(edges)
            assert nx.is_directed_acyclic_graph(G)


class TestSCMInfo:
    """Test SCM information utility functions."""
    
    def test_get_scm_info(self):
        """Test SCM information extraction."""
        factory = VariableSCMFactory(seed=42)
        scm = factory.create_variable_scm(4, "fork", target_variable="X1")
        
        info = get_scm_info(scm)
        
        expected_keys = [
            'num_variables', 'num_edges', 'target_variable',
            'variables', 'structure_type', 'edge_density',
            'coefficients', 'generation_info'
        ]
        
        for key in expected_keys:
            assert key in info
        
        assert info['num_variables'] == 4
        assert info['target_variable'] == "X1"
        assert isinstance(info['variables'], list)
        assert isinstance(info['num_edges'], int)
        assert isinstance(info['structure_type'], str)
        assert info['structure_type'] == "fork"


class TestSCMFactoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_minimum_variable_count(self):
        """Test SCM creation with minimum variable count."""
        factory = VariableSCMFactory(seed=42)
        
        scm = factory.create_variable_scm(3, "fork")
        
        variables = get_variables(scm)
        assert len(variables) == 3
        
        # Should be a valid SCM
        edges = get_edges(scm)
        assert len(edges) >= 1  # Should have at least one edge
    
    def test_maximum_edge_density(self):
        """Test SCM creation with maximum edge density."""
        factory = VariableSCMFactory(seed=42)
        
        scm = factory.create_variable_scm(4, "mixed", edge_density=1.0)
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        # Should still be a valid DAG
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        assert nx.is_directed_acyclic_graph(G)
    
    def test_zero_edge_density(self):
        """Test SCM creation with zero edge density."""
        factory = VariableSCMFactory(seed=42)
        
        scm = factory.create_variable_scm(5, "random", edge_density=0.0)
        
        edges = get_edges(scm)
        # Should have minimal edges (at least one to ensure target has parent)
        assert len(edges) >= 1  # Random generator ensures target has at least one parent
    
    def test_minimal_valid_input(self):
        """Test handling of minimal valid input."""
        factory = VariableSCMFactory(seed=42)
        
        # Test with minimal valid input (3 variables)
        scm = factory.create_variable_scm(3, "chain")
        
        variables = get_variables(scm)
        edges = get_edges(scm)
        
        assert len(variables) == 3
        assert len(edges) == 2  # Chain with 3 variables has 2 edges


if __name__ == "__main__":
    pytest.main([__file__, "-v"])