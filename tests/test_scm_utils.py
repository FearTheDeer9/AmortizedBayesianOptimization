"""
Tests for the utility methods and visualization capabilities of the SCM class.
"""
import unittest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import tempfile
import json

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.causal_graph import CausalGraph


# Define equation functions outside the class so they can be pickled
def x_equation():
    return 1


def y_equation(X):
    return 2 * X


def z_equation(X, Y):
    return X + Y


class TestSCMUtilities(unittest.TestCase):
    """Test cases for the utility methods of the StructuralCausalModel class."""

    def setUp(self):
        """Set up a test SCM with a simple structure."""
        # Create a simple SCM for testing
        self.scm = StructuralCausalModel(random_state=42)

        # Add variables
        self.scm.add_variable('X')
        self.scm.add_variable('Y')
        self.scm.add_variable('Z')

        # Define relationships
        self.scm.define_causal_relationship('Y', ['X'])
        self.scm.define_causal_relationship('Z', ['X', 'Y'])

        # Define deterministic equations for testing using the global functions
        self.scm.define_deterministic_equation('X', x_equation)  # Exogenous
        self.scm.define_deterministic_equation('Y', y_equation)  # Y = 2X
        self.scm.define_deterministic_equation('Z', z_equation)  # Z = X + Y

    def test_to_networkx(self):
        """Test conversion to NetworkX graph."""
        # Convert SCM to NetworkX graph
        G = self.scm.to_networkx()

        # Check that it's a directed graph
        self.assertIsInstance(G, nx.DiGraph)

        # Check nodes
        self.assertEqual(set(G.nodes()), {'X', 'Y', 'Z'})

        # Check edges
        expected_edges = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
        self.assertEqual(set(G.edges()), set(expected_edges))

        # Check node attributes
        for node in G.nodes():
            self.assertIn('has_equation', G.nodes[node])
            # All nodes have equations
            self.assertTrue(G.nodes[node]['has_equation'])

    def test_plot(self):
        """Test plotting functionality."""
        # Just check that plotting doesn't raise errors
        fig, ax = plt.subplots()
        try:
            self.scm.plot(ax=ax)
            plt.close(fig)  # Close to avoid displaying
        except Exception as e:
            self.fail(f"Plot method raised an exception: {e}")

        # Test with interventions
        self.scm.do_intervention('X', 2)
        fig, ax = plt.subplots()
        try:
            self.scm.plot(ax=ax, highlight_interventions=True)
            plt.close(fig)  # Close to avoid displaying
        except Exception as e:
            self.fail(
                f"Plot method with interventions raised an exception: {e}")

    def test_effect_calculation(self):
        """Test calculation of causal effects."""
        # The SCM is deterministic, so we can check exact values
        # Y = 2X, Z = X + Y = X + 2X = 3X

        # Set X = 1 (baseline) to X = 2 (intervention)
        # Expected change in Y: from 2*1=2 to 2*2=4, effect = 2
        direct_effect_on_y = self.scm.compute_direct_effect(
            treatment='X', outcome='Y', treatment_value=2, baseline_value=2
        )
        self.assertAlmostEqual(direct_effect_on_y, 2.0)

        # Expected change in Z: from 1+2=3 to 2+4=6, effect = 3
        total_effect_on_z = self.scm.compute_effect(
            treatment='X', outcome='Z', treatment_value=2, baseline_value=3
        )
        self.assertAlmostEqual(total_effect_on_z, 3.0)

        # Calculate indirect effect (should be total - direct)
        indirect_effect = self.scm.compute_indirect_effect(
            treatment='X', outcome='Z', treatment_value=2, baseline_value=3
        )

        # Direct effect of X on Z is just the X term in X + Y
        # So indirect effect should be through Y's change
        self.assertIsNotNone(indirect_effect)

    def test_save_load_json(self):
        """Test saving and loading SCM structure to/from JSON."""
        # Get JSON representation
        json_str = self.scm.to_json()

        # Check that it contains expected fields
        data = json.loads(json_str)
        self.assertIn('variables', data)
        self.assertIn('edges', data)
        self.assertIn('has_equations', data)

        # Check variables
        self.assertEqual(set(data['variables']), {'X', 'Y', 'Z'})

        # Check edges
        edges = [(edge['from'], edge['to']) for edge in data['edges']]
        expected_edges = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
        self.assertEqual(set(edges), set(expected_edges))

        # Test saving to file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.scm.to_json(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # Check file content
            with open(tmp_path, 'r') as f:
                file_content = f.read()

            self.assertEqual(json_str, file_content)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_load_pickle(self):
        """Test saving and loading complete SCM using pickle."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save SCM
            self.scm.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # Load SCM
            loaded_scm = StructuralCausalModel.load(tmp_path)

            # Check that it's the same type
            self.assertIsInstance(loaded_scm, StructuralCausalModel)

            # Check variables
            self.assertEqual(loaded_scm._variable_names,
                             self.scm._variable_names)

            # Check that equations work
            sample1 = self.scm.sample_data(1)
            sample2 = loaded_scm.sample_data(1)
            self.assertEqual(sample1['Z'].values[0], sample2['Z'].values[0])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_compare_structure(self):
        """Test comparing SCM structures."""
        # Create an identical SCM
        identical_scm = StructuralCausalModel(random_state=42)
        identical_scm.add_variable('X')
        identical_scm.add_variable('Y')
        identical_scm.add_variable('Z')
        identical_scm.define_causal_relationship('Y', ['X'])
        identical_scm.define_causal_relationship('Z', ['X', 'Y'])

        # Compare with itself (should be equivalent)
        is_equivalent, details = self.scm.compare_structure(identical_scm)
        self.assertTrue(is_equivalent)

        # Create a different SCM
        different_scm = StructuralCausalModel(random_state=42)
        different_scm.add_variable('X')
        different_scm.add_variable('Y')
        different_scm.add_variable('Z')
        different_scm.define_causal_relationship('Y', ['X'])
        different_scm.define_causal_relationship(
            'Z', ['Y'])  # Different relationship

        # Compare with different SCM (should not be equivalent)
        is_equivalent, details = self.scm.compare_structure(different_scm)
        self.assertFalse(is_equivalent)
        self.assertIn('edge_diff', details)

        # Create SCM with different variables
        missing_var_scm = StructuralCausalModel(random_state=42)
        missing_var_scm.add_variable('X')
        missing_var_scm.add_variable('Y')

        # Compare with missing variable SCM (should not be equivalent)
        is_equivalent, details = self.scm.compare_structure(missing_var_scm)
        self.assertFalse(is_equivalent)
        self.assertIn('variable_diff', details)

    def test_string_representation(self):
        """Test string representation methods."""
        # Test __str__
        str_repr = str(self.scm)
        self.assertIn('StructuralCausalModel with 3 variables', str_repr)
        self.assertIn('X', str_repr)
        self.assertIn('Y', str_repr)
        self.assertIn('Z', str_repr)

        # Test with interventions
        self.scm.do_intervention('X', 2)
        str_with_intervention = str(self.scm)
        self.assertIn('Active Interventions', str_with_intervention)
        self.assertIn('do(X = 2)', str_with_intervention)

        # Test __repr__
        repr_str = repr(self.scm)
        self.assertIn('StructuralCausalModel with 3 variables', repr_str)
        self.assertIn('3 equations', repr_str)


if __name__ == '__main__':
    unittest.main()
