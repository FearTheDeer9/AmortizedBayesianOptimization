"""
Tests for the Structural Causal Model (SCM) implementation.
"""
import unittest
import numpy as np
import pandas as pd

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.causal_graph import CausalGraph


class TestStructuralCausalModel(unittest.TestCase):
    """Test cases for the StructuralCausalModel class."""

    def setUp(self):
        """Set up test case with a simple SCM."""
        # Create a simple SCM for testing
        self.scm = StructuralCausalModel(random_state=42)

    def test_initialization(self):
        """Test that SCM initialization works correctly."""
        # Test with default parameters
        scm = StructuralCausalModel()
        self.assertEqual(scm._variable_names, [])
        self.assertEqual(scm._variable_domains, {})
        self.assertIsNone(scm._causal_graph)

        # Test with provided parameters
        causal_graph = CausalGraph()
        variable_names = ['X', 'Y', 'Z']
        variable_domains = {'X': 'binary',
                            'Y': 'continuous', 'Z': 'categorical'}

        for var in variable_names:
            causal_graph.add_node(var)

        causal_graph.add_edge('X', 'Y')
        causal_graph.add_edge('X', 'Z')
        causal_graph.add_edge('Y', 'Z')

        scm = StructuralCausalModel(
            causal_graph=causal_graph,
            variable_names=variable_names,
            variable_domains=variable_domains,
            random_state=42
        )

        self.assertEqual(scm._variable_names, variable_names)
        self.assertEqual(scm._variable_domains, variable_domains)
        self.assertEqual(scm._causal_graph, causal_graph)

    def test_add_variable(self):
        """Test adding variables to the SCM."""
        # Add variables
        self.scm.add_variable('X', domain='binary')
        self.scm.add_variable('Y', domain='continuous',
                              metadata={'min': 0, 'max': 10})
        self.scm.add_variable('Z', domain='categorical', metadata={
                              'categories': ['a', 'b', 'c']})

        # Check that variables were added correctly
        self.assertEqual(self.scm._variable_names, ['X', 'Y', 'Z'])
        self.assertEqual(self.scm._variable_domains, {
            'X': 'binary',
            'Y': 'continuous',
            'Z': 'categorical'
        })
        self.assertEqual(self.scm._variable_metadata['Y'], {
                         'min': 0, 'max': 10})
        self.assertEqual(self.scm._variable_metadata['Z'], {
                         'categories': ['a', 'b', 'c']})

        # Test adding a duplicate variable (should raise ValueError)
        with self.assertRaises(ValueError):
            self.scm.add_variable('X')

    def test_define_causal_relationship(self):
        """Test defining causal relationships between variables."""
        # Add variables
        self.scm.add_variable('X')
        self.scm.add_variable('Y')
        self.scm.add_variable('Z')

        # Define relationships
        self.scm.define_causal_relationship('Y', ['X'])
        self.scm.define_causal_relationship('Z', ['X', 'Y'])

        # Check that relationships were added correctly
        self.assertEqual(set(self.scm.get_parents('Y')), {'X'})
        self.assertEqual(set(self.scm.get_parents('Z')), {'X', 'Y'})
        self.assertEqual(set(self.scm.get_children('X')), {'Y', 'Z'})
        self.assertEqual(set(self.scm.get_children('Y')), {'Z'})

        # Test defining a relationship with a non-existent variable
        with self.assertRaises(ValueError):
            self.scm.define_causal_relationship('W', ['X'])

        # Test defining a cyclic relationship
        with self.assertRaises(ValueError):
            self.scm.define_causal_relationship('X', ['Z'])

    def test_validate_acyclicity(self):
        """Test that acyclicity validation works correctly."""
        # Create a simple causal graph
        scm = StructuralCausalModel()
        scm.add_variable('A')
        scm.add_variable('B')
        scm.add_variable('C')

        # This should be acyclic
        scm.define_causal_relationship('B', ['A'])
        scm.define_causal_relationship('C', ['B'])
        self.assertTrue(scm._validate_acyclicity())

        # This should raise an error due to the cycle
        with self.assertRaises(ValueError):
            scm.define_causal_relationship('A', ['C'])

    def test_get_parents_children(self):
        """Test getting parents and children of variables."""
        # Create a more complex causal graph
        scm = StructuralCausalModel()
        for var in ['A', 'B', 'C', 'D', 'E']:
            scm.add_variable(var)

        # Define relationships
        scm.define_causal_relationship('B', ['A'])
        scm.define_causal_relationship('C', ['A'])
        scm.define_causal_relationship('D', ['B', 'C'])
        scm.define_causal_relationship('E', ['C'])

        # Test get_parents
        self.assertEqual(scm.get_parents('A'), [])
        self.assertEqual(set(scm.get_parents('B')), {'A'})
        self.assertEqual(set(scm.get_parents('C')), {'A'})
        self.assertEqual(set(scm.get_parents('D')), {'B', 'C'})
        self.assertEqual(set(scm.get_parents('E')), {'C'})

        # Test get_children
        self.assertEqual(set(scm.get_children('A')), {'B', 'C'})
        self.assertEqual(set(scm.get_children('B')), {'D'})
        self.assertEqual(set(scm.get_children('C')), {'D', 'E'})
        self.assertEqual(scm.get_children('D'), [])
        self.assertEqual(scm.get_children('E'), [])

        # Test with non-existent variable
        with self.assertRaises(ValueError):
            scm.get_parents('F')
        with self.assertRaises(ValueError):
            scm.get_children('F')

    def test_define_deterministic_equation(self):
        """Test defining deterministic structural equations."""
        # Create SCM with variables and relationships
        scm = StructuralCausalModel(random_state=42)
        scm.add_variable('X')
        scm.add_variable('Y')
        scm.add_variable('Z')

        scm.define_causal_relationship('Y', ['X'])
        scm.define_causal_relationship('Z', ['X', 'Y'])

        # Define deterministic equation for Y (Y = 2*X + 1)
        def y_equation(X):
            return 2 * X + 1

        scm.define_deterministic_equation('Y', y_equation)

        # Define deterministic equation for Z (Z = X + Y)
        def z_equation(X, Y):
            return X + Y

        scm.define_deterministic_equation('Z', z_equation)

        # Check that the equations were stored correctly
        self.assertIn('Y', scm._structural_equations)
        self.assertIn('Z', scm._structural_equations)

        # Test evaluating the equations
        y_value = scm.evaluate_equation('Y', {'X': 2})
        self.assertEqual(y_value, 5)  # Y = 2*2 + 1 = 5

        z_value = scm.evaluate_equation('Z', {'X': 2, 'Y': 5})
        self.assertEqual(z_value, 7)  # Z = 2 + 5 = 7

        # Test evaluating with missing parent values
        with self.assertRaises(ValueError):
            scm.evaluate_equation('Z', {'X': 2})

    def test_define_probabilistic_equation(self):
        """Test defining probabilistic structural equations."""
        # Create SCM with variables and relationships
        scm = StructuralCausalModel(random_state=42)
        scm.add_variable('X')
        scm.add_variable('Y')

        scm.define_causal_relationship('Y', ['X'])

        # Define a probabilistic equation for Y (Y = X + noise)
        def y_equation(X, noise):
            return X + noise

        # Define noise distribution
        def noise_dist():
            return np.random.normal(0, 1)

        scm.define_probabilistic_equation('Y', y_equation, noise_dist)

        # Check that the equation and noise function were stored correctly
        self.assertIn('Y', scm._structural_equations)
        self.assertIn('Y', scm._exogenous_functions)

        # Test invalid probabilistic equation (no noise parameter)
        def invalid_equation(X):
            return X

        with self.assertRaises(ValueError):
            scm.define_probabilistic_equation(
                'Y', invalid_equation, noise_dist)

    def test_define_linear_gaussian_equation(self):
        """Test defining linear Gaussian structural equations."""
        # Create SCM with variables and relationships
        scm = StructuralCausalModel(random_state=42)
        scm.add_variable('X')
        scm.add_variable('Y')
        scm.add_variable('Z')

        scm.define_causal_relationship('Y', ['X'])
        scm.define_causal_relationship('Z', ['X', 'Y'])

        # Define linear Gaussian equation for Y
        scm.define_linear_gaussian_equation(
            'Y', {'X': 2.0}, intercept=1.0, noise_std=0.5)

        # Define linear Gaussian equation for Z
        scm.define_linear_gaussian_equation(
            'Z', {'X': 1.0, 'Y': 1.5}, intercept=-0.5, noise_std=0.0)

        # Check that the equations were stored correctly
        self.assertIn('Y', scm._structural_equations)
        self.assertIn('Y', scm._exogenous_functions)
        self.assertIn('Z', scm._structural_equations)
        # No noise function for Z
        self.assertNotIn('Z', scm._exogenous_functions)

        # Test evaluating equation for Z (should be deterministic)
        z_value = scm.evaluate_equation('Z', {'X': 2, 'Y': 3})
        # Z = 1.0*X + 1.5*Y - 0.5 = 2 + 4.5 - 0.5 = 6.0
        self.assertEqual(z_value, 2 + 3*1.5 - 0.5)

        # Test invalid coefficient
        with self.assertRaises(ValueError):
            scm.define_linear_gaussian_equation('Y', {'W': 1.0})

    def test_evaluate_equation(self):
        """Test evaluating structural equations."""
        # Create SCM with variables, relationships, and equations
        scm = StructuralCausalModel(random_state=42)
        scm.add_variable('X')
        scm.add_variable('Y')
        scm.add_variable('Z')

        scm.define_causal_relationship('Y', ['X'])
        scm.define_causal_relationship('Z', ['X', 'Y'])

        # Define deterministic equation for Y
        def y_equation(X):
            return X ** 2

        scm.define_deterministic_equation('Y', y_equation)

        # Define linear Gaussian equation for Z
        scm.define_linear_gaussian_equation(
            'Z', {'X': 0.5, 'Y': 1.0}, intercept=0.0, noise_std=0.0)

        # Test evaluating equations with different input values
        test_cases = [
            {'X': 0, 'expected_Y': 0, 'expected_Z': 0},
            {'X': 1, 'expected_Y': 1, 'expected_Z': 1.5},
            {'X': 2, 'expected_Y': 4, 'expected_Z': 5.0},
            {'X': -1, 'expected_Y': 1, 'expected_Z': 0.5},
        ]

        for case in test_cases:
            x_val = case['X']
            y_val = scm.evaluate_equation('Y', {'X': x_val})
            z_val = scm.evaluate_equation('Z', {'X': x_val, 'Y': y_val})

            self.assertEqual(y_val, case['expected_Y'])
            self.assertEqual(z_val, case['expected_Z'])

        # Test evaluating equation for variable without defined equation
        with self.assertRaises(ValueError):
            scm.evaluate_equation('X', {})


if __name__ == '__main__':
    unittest.main()
