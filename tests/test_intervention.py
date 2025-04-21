"""
Tests for the intervention methods in the Structural Causal Model.
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.causal_graph import CausalGraph


class TestInterventions(unittest.TestCase):
    """Test cases for intervention functionality in the StructuralCausalModel class."""

    def setUp(self):
        """Set up the test case with a simple SCM for interventions."""
        # Create a simple SCM for testing
        self.scm = StructuralCausalModel(random_state=42)

        # Add variables
        self.scm.add_variable('X')
        self.scm.add_variable('Y')
        self.scm.add_variable('Z')

        # Define relationships
        self.scm.define_causal_relationship('Y', ['X'])
        self.scm.define_causal_relationship('Z', ['X', 'Y'])

        # Define equations
        def x_equation():
            return np.random.normal(0, 1)

        def y_equation(X):
            return 2 * X + 1

        def z_equation(X, Y):
            return X + Y

        self.scm.define_structural_equation('X', x_equation)
        self.scm.define_deterministic_equation('Y', y_equation)
        self.scm.define_deterministic_equation('Z', z_equation)

    @patch.object(StructuralCausalModel, 'sample_data')
    def test_do_intervention(self, mock_sample_data):
        """Test do_intervention method."""
        # Set up the mock to return a test DataFrame
        mock_df = pd.DataFrame({
            'X': [3, 3, 3],
            'Y': [1, 1, 1],
            'Z': [4, 4, 4]
        })
        mock_sample_data.return_value = mock_df

        # Verify initial state has no interventions
        self.assertFalse(hasattr(self.scm, '_original_graph'))
        self.assertEqual(self.scm._interventions, {})

        # Perform an intervention
        self.scm.do_intervention('X', 3)

        # Verify intervention was recorded
        self.assertEqual(self.scm._interventions, {'X': 3})

        # Verify original graph was stored
        self.assertTrue(hasattr(self.scm, '_original_graph'))

        # Check that parents of X are removed in the modified graph
        self.assertEqual(self.scm._causal_graph.get_parents('X'), set())

        # Test intervening on a non-existent variable
        with self.assertRaises(ValueError):
            self.scm.do_intervention('W', 5)

    def test_reset(self):
        """Test reset method."""
        # First do an intervention
        self.scm.do_intervention('X', 3)

        # Verify intervention was recorded
        self.assertEqual(self.scm._interventions, {'X': 3})

        # Reset the SCM
        self.scm.reset()

        # Verify interventions were cleared
        self.assertEqual(self.scm._interventions, {})

        # Verify graph was restored
        self.assertEqual(self.scm._causal_graph.get_node_attributes('X'), {})

        # Reset should work even if no interventions were done
        new_scm = StructuralCausalModel()
        new_scm.reset()  # This should not cause any errors

    @patch.object(StructuralCausalModel, 'sample_data')
    def test_multiple_interventions(self, mock_sample_data):
        """Test multiple_interventions method."""
        # Set up the mock to return a test DataFrame
        mock_df = pd.DataFrame({
            'X': [2, 2, 2],
            'Y': [5, 5, 5],
            'Z': [7, 7, 7]
        })
        mock_sample_data.return_value = mock_df

        # Perform multiple interventions
        interventions = {'X': 2, 'Y': 5}
        self.scm.multiple_interventions(interventions)

        # Verify interventions were recorded
        self.assertEqual(self.scm._interventions, interventions)

        # Check that parents are removed for both intervened variables
        self.assertEqual(self.scm._causal_graph.get_parents('X'), set())
        self.assertEqual(self.scm._causal_graph.get_parents('Y'), set())

        # Test with non-existent variable
        with self.assertRaises(ValueError):
            self.scm.multiple_interventions({'W': 5})

    @patch.object(StructuralCausalModel, 'sample_data')
    def test_sample_interventional_data(self, mock_sample_data):
        """Test sample_interventional_data method."""
        # Set up the mock to return a test DataFrame
        mock_df = pd.DataFrame({
            'X': [2, 2, 2],
            'Y': [5, 5, 5],
            'Z': [7, 7, 7]
        })
        mock_sample_data.return_value = mock_df

        # Sample data with interventions
        interventions = {'X': 2}
        result = self.scm.sample_interventional_data(
            interventions, sample_size=3)

        # Verify the result
        pd.testing.assert_frame_equal(result, mock_df)

        # Verify that mock_sample_data was called with the right parameters
        mock_sample_data.assert_called_once_with(3, None)

        # Verify that state was restored (no active interventions)
        self.assertEqual(self.scm._interventions, {})

        # Test with multiple interventions
        mock_sample_data.reset_mock()
        interventions = {'X': 2, 'Y': 5}
        result = self.scm.sample_interventional_data(
            interventions, sample_size=5, random_seed=42)

        # Verify the results again
        pd.testing.assert_frame_equal(result, mock_df)

        # Verify that mock_sample_data was called with the right parameters
        mock_sample_data.assert_called_once_with(5, 42)

    @patch.object(StructuralCausalModel, 'sample_data')
    def test_get_intervention_effects(self, mock_sample_data):
        """Test get_intervention_effects method."""
        # Set up the mock to return different test DataFrames for different interventions
        def sample_data_side_effect(sample_size, random_seed):
            if self.scm._interventions.get('X') == 2:
                return pd.DataFrame({
                    'X': [2, 2, 2],
                    'Y': [5, 5, 5],
                    'Z': [7, 7, 7]
                })
            elif self.scm._interventions.get('X') == 3:
                return pd.DataFrame({
                    'X': [3, 3, 3],
                    'Y': [7, 7, 7],
                    'Z': [10, 10, 10]
                })
            else:
                return pd.DataFrame({
                    'X': [0, 0, 0],
                    'Y': [1, 1, 1],
                    'Z': [1, 1, 1]
                })

        mock_sample_data.side_effect = sample_data_side_effect

        # Calculate intervention effects
        intervention_values = [0, 2, 3]
        effects = self.scm.get_intervention_effects(
            'X', 'Z', intervention_values, sample_size=3)

        # Verify the results
        expected_effects = {0: 1.0, 2: 7.0, 3: 10.0}
        self.assertEqual(effects, expected_effects)

        # Verify that state was restored (no active interventions)
        self.assertEqual(self.scm._interventions, {})

        # Test with invalid variables
        with self.assertRaises(ValueError):
            self.scm.get_intervention_effects('W', 'Z', [1, 2])

        with self.assertRaises(ValueError):
            self.scm.get_intervention_effects('X', 'W', [1, 2])


if __name__ == '__main__':
    unittest.main()
