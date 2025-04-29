import pytest
import numpy as np
import pandas as pd
from numpy.random import RandomState
import copy
import unittest

from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.environments.interventions import PerfectIntervention, ImperfectIntervention, SoftIntervention

# Fixture for a simple SCM: X -> Y -> Z
@pytest.fixture
def simple_linear_scm():
    # Define graph
    cg = CausalGraph()
    # Add nodes FIRST
    cg.add_node("X")
    cg.add_node("Y")
    cg.add_node("Z")
    # Then add edges
    cg.add_edge("X", "Y")
    cg.add_edge("Y", "Z")

    # Define SCM
    # Pass the graph with nodes and edges already defined
    scm = StructuralCausalModel(causal_graph=cg)
    # Explicitly add variables to the SCM instance *after* initialization
    scm.add_variable("X")
    scm.add_variable("Y")
    scm.add_variable("Z")

    # Define linear equations with Gaussian noise
    scm.define_linear_gaussian_equation("X", {}, intercept=0, noise_std=1.0) # Exogenous X
    scm.define_linear_gaussian_equation("Y", {"X": 2.0}, intercept=1.0, noise_std=0.5) # Y = 2*X + 1 + N_Y
    scm.define_linear_gaussian_equation("Z", {"Y": -1.0}, intercept=0.5, noise_std=0.2) # Z = -Y + 0.5 + N_Z

    return scm

# =================== Tests for sample_data_interventional ===================

def test_sample_interventional_perfect(simple_linear_scm):
    """Test sampling with a perfect intervention."""
    intervention_value = 10.0
    intervention = PerfectIntervention(target_node="Y", value=intervention_value)
    sample_size = 50
    original_scm_copy = copy.deepcopy(simple_linear_scm) # For checking immutability

    data = simple_linear_scm.sample_data_interventional(
        interventions=[intervention],
        sample_size=sample_size,
        random_seed=123
    )

    assert isinstance(data, pd.DataFrame)
    assert len(data) == sample_size
    assert list(data.columns) == ["X", "Y", "Z"]
    # Convert to numeric before comparison
    y_numeric = pd.to_numeric(data["Y"])
    z_numeric = pd.to_numeric(data["Z"])
    # Check if Y is fixed to the intervention value
    assert np.allclose(y_numeric, intervention_value)
    # Check if Z depends on the intervened Y
    # Z = -Y + 0.5 + N_Z => Z should be around -10.0 + 0.5 = -9.5
    assert abs(z_numeric.mean() - (-intervention_value + 0.5)) < 0.5 # Allow for noise

    # Verify original SCM is unchanged (basic check)
    assert simple_linear_scm._structural_equations["Y"] == original_scm_copy._structural_equations["Y"]
    assert simple_linear_scm.get_causal_graph().get_parents("Y") == original_scm_copy.get_causal_graph().get_parents("Y")

def test_sample_interventional_imperfect_additive(simple_linear_scm):
    """Test sampling with an additive imperfect intervention."""
    intervention_val = 5.0
    strength = 0.8
    intervention = ImperfectIntervention(
        target_node="Y",
        value=intervention_val,
        strength=strength,
        combination_method='additive'
    )
    sample_size = 200 # Larger sample for mean check

    data = simple_linear_scm.sample_data_interventional(
        interventions=[intervention],
        sample_size=sample_size,
        random_seed=456
    )

    assert len(data) == sample_size
    # Original Y = 2*X + 1 + N_Y
    # Intervened Y = (2*X + 1 + N_Y) + strength * intervention_val
    # E[X] ~ 0, E[N_Y] ~ 0 => E[Original Y] ~ 1
    # E[Intervened Y] ~ E[Original Y] + strength * intervention_val
    # E[Intervened Y] ~ 1 + 0.8 * 5.0 = 1 + 4.0 = 5.0
    assert abs(data["Y"].mean() - (1.0 + strength * intervention_val)) < 0.5 # Check mean allows for noise

def test_sample_interventional_soft_replace(simple_linear_scm):
    """Test sampling with a soft intervention replacing the mechanism."""
    def new_y_mechanism(X, **kwargs): # Accepts parent X
        return X**2 + 3.0 # New quadratic relationship

    intervention = SoftIntervention(
        target_node="Y",
        intervention_function=new_y_mechanism,
        compose=False
    )
    sample_size = 100

    data = simple_linear_scm.sample_data_interventional(
        interventions=[intervention],
        sample_size=sample_size,
        random_seed=789
    )

    assert len(data) == sample_size
    # Convert to numeric before comparison
    x_numeric = pd.to_numeric(data["X"])
    y_numeric = pd.to_numeric(data["Y"])
    # Check if Y follows the new mechanism Y = X^2 + 3.0 (approximately)
    expected_y = x_numeric**2 + 3.0
    assert np.allclose(y_numeric, expected_y, atol=1e-6) # Should be deterministic if no noise in new func

def test_sample_interventional_multiple(simple_linear_scm):
    """Test sampling with multiple interventions applied sequentially."""
    interv1_val = 5.0
    interv1 = PerfectIntervention(target_node="X", value=interv1_val) # Fix X
    interv2 = ImperfectIntervention(target_node="Y", value=10.0, strength=0.5, combination_method='weighted_average') # Modify Y

    sample_size = 100
    data = simple_linear_scm.sample_data_interventional(
        interventions=[interv1, interv2],
        sample_size=sample_size,
        random_seed=101
    )

    assert len(data) == sample_size
    # Convert to numeric before comparison
    x_numeric = pd.to_numeric(data["X"])
    y_numeric = pd.to_numeric(data["Y"])
    # Check X is fixed
    assert np.allclose(x_numeric, interv1_val)
    # Check Y is modified based on fixed X and imperfect intervention
    # Original Y = 2*X + 1 + N_Y => 2*5 + 1 + N_Y = 11 + N_Y
    # Intervened Y = (1-strength)*OriginalY + strength*IntervVal
    # E[Intervened Y] = (1-0.5)*(11) + 0.5 * 10.0 = 0.5*11 + 5 = 5.5 + 5 = 10.5
    assert abs(y_numeric.mean() - 10.5) < 0.5

def test_sample_interventional_reproducibility(simple_linear_scm):
    """Test reproducibility with random_seed."""
    intervention = PerfectIntervention(target_node="Y", value=5.0)
    sample_size = 10
    seed = 999

    data1 = simple_linear_scm.sample_data_interventional(
        interventions=[intervention], sample_size=sample_size, random_seed=seed)
    data2 = simple_linear_scm.sample_data_interventional(
        interventions=[intervention], sample_size=sample_size, random_seed=seed)

    pd.testing.assert_frame_equal(data1, data2)

def test_sample_interventional_invalid_input(simple_linear_scm):
    """Test error handling for invalid intervention inputs."""
    with pytest.raises(TypeError, match="must be a list"):
        simple_linear_scm.sample_data_interventional(interventions=None, sample_size=10)

    # Assuming Intervention class is available for this check
    # with pytest.raises(TypeError, match="must be Intervention objects"):
    #     simple_linear_scm.sample_data_interventional(interventions=[1, 2], sample_size=10)

    bad_intervention = PerfectIntervention(target_node="W", value=1) # W doesn't exist
    with pytest.raises(ValueError, match="Failed to apply intervention"):
        simple_linear_scm.sample_data_interventional(interventions=[bad_intervention], sample_size=10)

# =================== Other SCM Tests (Add as needed) =======================

def test_get_adjacency_matrix(simple_linear_scm):
    """Test getting the adjacency matrix."""
    scm = simple_linear_scm
    adj_matrix = scm.get_adjacency_matrix()

    # Expected matrix for X -> Y -> Z (order might vary if not specified)
    # Assuming default order [X, Y, Z]
    expected_adj = np.array([
        [0, 1, 0],  # X row: edge to Y
        [0, 0, 1],  # Y row: edge to Z
        [0, 0, 0]   # Z row: no outgoing edges
    ])

    # Get adj matrix with specified node order
    node_order = ["X", "Y", "Z"]
    adj_matrix_ordered = scm.get_adjacency_matrix(node_order=node_order)

    assert isinstance(adj_matrix_ordered, np.ndarray)
    assert adj_matrix_ordered.shape == (3, 3)
    np.testing.assert_array_equal(adj_matrix_ordered, expected_adj)

    # Test with a different node order
    node_order_rev = ["Z", "Y", "X"]
    expected_adj_rev = np.array([
        [0, 0, 0],  # Z row
        [1, 0, 0],  # Y row: edge to Z (now at index 0)
        [0, 1, 0]   # X row: edge to Y (now at index 1)
    ])
    # Check SCM's method directly with the new order
    adj_matrix_rev = scm.get_adjacency_matrix(node_order=node_order_rev)
    if hasattr(adj_matrix_rev, 'toarray'): # Handle sparse matrix case if needed
        adj_matrix_rev = adj_matrix_rev.toarray()
    np.testing.assert_array_equal(adj_matrix_rev, expected_adj_rev)

    # Test case where graph is None (should raise ValueError)
    scm_no_graph = StructuralCausalModel()
    with pytest.raises(ValueError, match="Causal graph is not defined"):
        scm_no_graph.get_adjacency_matrix()

def test_scm_initialization(simple_linear_scm):
    assert simple_linear_scm is not None
    assert set(simple_linear_scm.get_variable_names()) == {"X", "Y", "Z"}
    assert simple_linear_scm.get_causal_graph() is not None

def test_sample_data_observational(simple_linear_scm):
    sample_size = 100
    data = simple_linear_scm.sample_data(sample_size=sample_size, random_seed=42)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == sample_size
    assert list(data.columns) == ["X", "Y", "Z"]
    # Add more specific checks if needed based on expected distributions 

class TestStructuralCausalModel(unittest.TestCase):
    def test_get_adjacency_matrix_simple_chain(self):
        """Test getting adjacency matrix for a simple chain graph."""
        scm = StructuralCausalModel(variable_names=['A', 'B', 'C'])
        scm.define_causal_relationship('B', ['A'])
        scm.define_causal_relationship('C', ['B'])
        
        adj_matrix = scm.get_adjacency_matrix(node_order=['A', 'B', 'C'])
        expected_matrix = np.array([
            [0, 1, 0],  # A -> B
            [0, 0, 1],  # B -> C
            [0, 0, 0]   # C -> _
        ])
        np.testing.assert_array_equal(adj_matrix, expected_matrix)

    def test_get_adjacency_matrix_fork(self):
        """Test getting adjacency matrix for a fork structure."""
        scm = StructuralCausalModel(variable_names=['A', 'B', 'C'])
        scm.define_causal_relationship('B', ['A'])
        scm.define_causal_relationship('C', ['A'])
        
        adj_matrix = scm.get_adjacency_matrix(node_order=['A', 'B', 'C'])
        expected_matrix = np.array([
            [0, 1, 1],  # A -> B, A -> C
            [0, 0, 0],  # B -> _
            [0, 0, 0]   # C -> _
        ])
        np.testing.assert_array_equal(adj_matrix, expected_matrix)

    def test_get_adjacency_matrix_collider(self):
        """Test getting adjacency matrix for a collider structure."""
        scm = StructuralCausalModel(variable_names=['A', 'B', 'C'])
        scm.define_causal_relationship('C', ['A', 'B'])
        
        adj_matrix = scm.get_adjacency_matrix(node_order=['A', 'B', 'C'])
        expected_matrix = np.array([
            [0, 0, 1],  # A -> C
            [0, 0, 1],  # B -> C
            [0, 0, 0]   # C -> _
        ])
        np.testing.assert_array_equal(adj_matrix, expected_matrix)

    def test_get_adjacency_matrix_no_graph(self):
        """Test getting adjacency matrix when no graph is defined."""
        scm = StructuralCausalModel(variable_names=['A', 'B'])
        # The current implementation raises ValueError, let's test that
        with self.assertRaises(ValueError):
             scm.get_adjacency_matrix()
             
    def test_get_adjacency_matrix_different_order(self):
        """Test getting adjacency matrix with a different node order."""
        scm = StructuralCausalModel(variable_names=['X', 'Y', 'Z'])
        scm.define_causal_relationship('Y', ['X'])
        scm.define_causal_relationship('Z', ['Y'])
        
        adj_matrix = scm.get_adjacency_matrix(node_order=['Z', 'X', 'Y'])
        # Z -> _, X -> Y, Y -> Z
        expected_matrix = np.array([
            [0, 0, 0],  # Z -> _
            [0, 0, 1],  # X -> Y
            [1, 0, 0]   # Y -> Z
        ])
        np.testing.assert_array_equal(adj_matrix, expected_matrix)

    # ... other existing tests ... 