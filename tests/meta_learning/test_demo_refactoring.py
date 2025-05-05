#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the refactored demo scripts using the Component Registry.

This test suite ensures that the refactored demos correctly use
the existing components from the causal_meta package instead of
duplicating functionality.
"""

import os
import sys
import pytest
import numpy as np
import torch
import importlib.util
from typing import Dict, Any
import matplotlib.pyplot as plt

# Add the root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import demo utilities
from demos import utils


def is_module_available(module_name: str) -> bool:
    """Check if a module is available."""
    try:
        importlib.util.find_spec(module_name)
        return True
    except ImportError:
        return False


class TestDemoUtilities:
    """Test the enhanced utility functions in demos/utils.py."""
    
    def test_safe_import(self):
        """Test the safe_import function."""
        # Test successful import
        np_module = utils.safe_import('numpy')
        assert np_module is not None
        assert np_module.__name__ == 'numpy'
        
        # Test class import
        CausalGraph = utils.safe_import('causal_meta.graph.causal_graph.CausalGraph')
        if is_module_available('causal_meta.graph.causal_graph'):
            assert CausalGraph is not None
        
        # Test fallback
        fallback = {'dummy': True}
        result = utils.safe_import('non_existent_module.class', fallback)
        assert result is fallback
    
    def test_tensor_shape_standardization(self):
        """Test tensor shape standardization for different components."""
        # Test 1D data for graph encoder
        data_1d = np.random.randn(10)
        result = utils.standardize_tensor_shape(data_1d, for_component='graph_encoder')
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 10, 1)
        
        # Test 2D data for graph encoder
        data_2d = np.random.randn(10, 5)
        result = utils.standardize_tensor_shape(data_2d, for_component='graph_encoder')
        assert result.shape == (1, 10, 5)
        
        # Test 2D data with custom batch size
        result = utils.standardize_tensor_shape(data_2d, for_component='graph_encoder', batch_size=3)
        assert result.shape == (3, 10, 5)
        
        # Test 2D data for dynamics decoder with proper shapes
        num_nodes = 5
        samples = 10
        # Create data with correct shape: [samples, features]
        decoder_data = np.random.randn(samples, 1)  # Shape matches [samples, 1]
        result = utils.standardize_tensor_shape(decoder_data, for_component='dynamics_decoder', num_nodes=num_nodes)
        # Check shape dimensions: expected result is [num_nodes * batch_size, feature_dim] = [5, feature_dim]
        assert result.shape[0] == num_nodes * 1  # num_nodes * batch_size 
        assert result.shape[1] >= 1  # Feature dimension exists
        
        # Test error when num_nodes is not provided for dynamics decoder
        with pytest.raises(ValueError):
            utils.standardize_tensor_shape(data_2d, for_component='dynamics_decoder')
        
        # Test unknown component type
        with pytest.raises(ValueError):
            utils.standardize_tensor_shape(data_2d, for_component='unknown_component')
    
    def test_node_name_conversion(self):
        """Test node name <-> ID conversion."""
        # Test get_node_name
        assert utils.get_node_name(0) == "X_0"
        assert utils.get_node_name(5) == "X_5"
        assert utils.get_node_name("X_3") == "X_3"
        
        # Test get_node_id
        assert utils.get_node_id(0) == 0
        assert utils.get_node_id("X_5") == 5
        
        # Test invalid node name
        with pytest.raises(ValueError):
            utils.get_node_id("invalid_name")
            
        with pytest.raises(ValueError):
            utils.get_node_id("X_invalid")
    
    def test_intervention_formatting(self):
        """Test intervention formatting."""
        interventions = {0: 1.0, "X_2": 2.0, 3: 3.0}
        
        # Test string formatting
        formatted = utils.format_interventions(interventions)
        assert isinstance(formatted, dict)
        assert "X_0" in formatted
        assert "X_2" in formatted
        assert "X_3" in formatted
        assert formatted["X_0"] == 1.0
        assert formatted["X_2"] == 2.0
        assert formatted["X_3"] == 3.0
        
        # Test tensor formatting
        tensor_format = utils.format_interventions(interventions, for_tensor=True, num_nodes=4)
        assert "targets" in tensor_format
        assert "values" in tensor_format
        assert isinstance(tensor_format["targets"], torch.Tensor)
        assert isinstance(tensor_format["values"], torch.Tensor)
        
        # Test tensor formatting with device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            tensor_format = utils.format_interventions(
                interventions, for_tensor=True, num_nodes=4, device=device
            )
            assert tensor_format["targets"].device.type == "cuda"
            assert tensor_format["values"].device.type == "cuda"
        
        # Test error when num_nodes is not provided
        with pytest.raises(ValueError):
            utils.format_interventions(interventions, for_tensor=True)
    
    def test_graph_creation(self):
        """Test creating CausalGraph from adjacency matrix."""
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Test default node names
        graph = utils.create_causal_graph_from_adjacency(adj_matrix)
        
        # Basic structure checks that should work with both real CausalGraph and DummyGraph
        assert hasattr(graph, "get_nodes")
        assert hasattr(graph, "get_edges")
        assert hasattr(graph, "get_adjacency_matrix")
        
        nodes = graph.get_nodes()
        assert len(nodes) == 3
        assert "X_0" in nodes
        assert "X_1" in nodes
        assert "X_2" in nodes
        
        edges = graph.get_edges()
        assert len(edges) == 2
        edge_pairs = [(e[0], e[1]) for e in edges]
        assert ("X_0", "X_1") in edge_pairs
        assert ("X_1", "X_2") in edge_pairs
        
        # Test custom node names
        node_names = ["A", "B", "C"]
        graph = utils.create_causal_graph_from_adjacency(adj_matrix, node_names=node_names)
        
        # Verify nodes without assuming order
        returned_nodes = graph.get_nodes()
        assert len(returned_nodes) == len(node_names)
        for node in node_names:
            assert node in returned_nodes
    
    def test_model_creation(self):
        """Test model creation utilities."""
        # Skip if causal_meta components are not available
        if not is_module_available('causal_meta.meta_learning.dynamics_decoder'):
            pytest.skip("causal_meta package not available")
        
        # Test creating a graph encoder
        model = utils.create_new_model('graph_encoder')
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test creating a dynamics decoder
        model = utils.create_new_model('dynamics_decoder')
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test creating an amortized causal discovery model
        model = utils.create_new_model('acd')
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test with custom parameters
        custom_params = {
            'hidden_dim': 128,
            'input_dim': 2,
            'num_layers': 3
        }
        model = utils.create_new_model('acd', model_params=custom_params)
        assert model is not None
        
        # Test with unknown model type
        model = utils.create_new_model('unknown_model_type')
        assert model is None


@pytest.mark.skipif(not is_module_available('causal_meta.graph.causal_graph'), 
                   reason="causal_meta package not available")
class TestDummyGraphFallback:
    """Test the DummyGraph implementation used as fallback."""
    
    def test_dummy_graph_basics(self):
        """Test basic functionality of DummyGraph."""
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        graph = utils.DummyGraph(adj_matrix)
        
        # Test node count
        assert graph.get_num_nodes() == 3
        
        # Test has_edge
        assert graph.has_edge(0, 1)
        assert graph.has_edge("X_0", "X_1")
        assert graph.has_edge(1, 2)
        assert not graph.has_edge(0, 2)
        assert not graph.has_edge(2, 0)
        
        # Test node listing
        assert graph.get_nodes() == ["X_0", "X_1", "X_2"]
        
        # Test edge listing
        edges = graph.get_edges()
        assert len(edges) == 2
        assert ("X_0", "X_1") in edges
        assert ("X_1", "X_2") in edges
        
        # Test adjacency matrix
        np.testing.assert_array_equal(graph.get_adjacency_matrix(), adj_matrix)
    
    def test_dummy_graph_node_operations(self):
        """Test node operations in DummyGraph."""
        adj_matrix = np.array([
            [0, 1],
            [0, 0]
        ])
        
        graph = utils.DummyGraph(adj_matrix)
        
        # Test adding a node
        graph.add_node("X_2")
        assert graph.get_num_nodes() == 3
        assert graph.get_nodes() == ["X_0", "X_1", "X_2"]
        
        # Test adjacency matrix expansion
        expected_adj = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        np.testing.assert_array_equal(graph.get_adjacency_matrix(), expected_adj)
        
        # Test adding an edge
        graph.add_edge("X_0", "X_2")
        expected_adj = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ])
        np.testing.assert_array_equal(graph.get_adjacency_matrix(), expected_adj)
        
        # Test adding a new node through edge addition
        graph.add_edge("X_3", "X_1")
        assert graph.get_num_nodes() == 4
        assert "X_3" in graph.get_nodes()
        assert graph.has_edge("X_3", "X_1")


class TestVisualizationUtilities:
    """Test the visualization utilities."""
    
    def test_fallback_plot_graph(self):
        """Test fallback graph plotting."""
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Create a figure and check that the function returns a valid axis
        fig, ax = plt.subplots()
        result_ax = utils.fallback_plot_graph(adj_matrix, ax=ax, title="Test Graph")
        
        assert result_ax is ax
        assert result_ax.get_title() == "Test Graph"
        
        # Clean up
        plt.close(fig)
    
    def test_visualize_graph(self):
        """Test the visualize_graph function."""
        adj_matrix = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        # Test with adjacency matrix
        fig, ax = plt.subplots()
        result_ax = utils.visualize_graph(adj_matrix, ax=ax, title="Test Graph")
        
        assert result_ax is ax
        
        # Test with DummyGraph
        graph = utils.DummyGraph(adj_matrix)
        fig, ax = plt.subplots()
        result_ax = utils.visualize_graph(graph, ax=ax, title="Dummy Graph")
        
        assert result_ax is ax
        
        # Test figure creation when ax is None
        result_ax = utils.visualize_graph(graph, title="Auto Figure")
        assert result_ax is not None
        
        # Clean up
        plt.close('all')
    
    def test_compare_graphs(self):
        """Test graph comparison function."""
        adj_matrix1 = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        adj_matrix2 = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        
        graph1 = utils.DummyGraph(adj_matrix1)
        graph2 = utils.DummyGraph(adj_matrix2)
        
        # Test comparison
        fig = utils.compare_graphs(graph1, graph2, "Graph 1", "Graph 2")
        
        assert fig is not None
        assert len(fig.axes) == 2
        
        # Clean up
        plt.close(fig)


class TestParentScaleAcdRefactoring:
    """Test the refactored parent_scale_acd_demo.py script."""
    
    def test_parent_scale_acd_imports(self):
        """Test that the demo properly imports components."""
        # Import the demo script
        from demos import parent_scale_acd_demo
        
        # Check that it's using safe imports for key components
        assert hasattr(parent_scale_acd_demo, 'GraphFactory')
        assert hasattr(parent_scale_acd_demo, 'StructuralCausalModel')
        assert hasattr(parent_scale_acd_demo, 'CausalGraph')
        assert hasattr(parent_scale_acd_demo, 'AmortizedCausalDiscovery')
    
    def test_utility_integration(self):
        """Test integration with utility functions."""
        # Import the demo script
        from demos import parent_scale_acd_demo
        
        # Test that it's using utility functions properly
        assert parent_scale_acd_demo.create_causal_graph_from_adjacency is not None
        assert parent_scale_acd_demo.format_interventions is not None
        assert parent_scale_acd_demo.standardize_tensor_shape is not None
        assert parent_scale_acd_demo.get_node_name is not None
        assert parent_scale_acd_demo.get_node_id is not None
        
    @pytest.mark.skipif(not is_module_available('causal_meta.meta_learning.amortized_causal_discovery'), 
                      reason="AmortizedCausalDiscovery not available")
    def test_model_enhancement(self):
        """Test model handling features."""
        from demos import parent_scale_acd_demo
        
        # Check that the model interfaces are properly defined
        assert hasattr(parent_scale_acd_demo.parent_scaled_acd, '__call__')
        
        # Verify the key method has the right parameters
        import inspect
        signature = inspect.signature(parent_scale_acd_demo.parent_scaled_acd)
        params = signature.parameters
        
        # Check for essential parameters
        assert 'model' in params
        assert 'scm' in params
        assert 'obs_data' in params
        assert 'device' in params


if __name__ == '__main__':
    pytest.main(['-xvs', __file__]) 