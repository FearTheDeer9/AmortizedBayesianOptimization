"""
Tests for the sampling functionality in the Structural Causal Model.
"""
import unittest
import numpy as np
import pandas as pd

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.graph.causal_graph import CausalGraph


class TestSampling(unittest.TestCase):
    """Test cases for sampling functionality in the StructuralCausalModel class."""

    def setUp(self):
        """Set up the test case with a simple SCM for sampling."""
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

    def test_sample_data(self):
        """Test that sample_data generates the expected number of samples and follows causal relationships."""
        # Generate samples
        sample_size = 100
        data = self.scm.sample_data(sample_size)

        # Check that the returned data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)

        # Check that the DataFrame has the correct columns
        self.assertEqual(set(data.columns), {'X', 'Y', 'Z'})

        # Check that the DataFrame has the correct number of rows
        self.assertEqual(len(data), sample_size)

        # Check that the relationships hold for each sample
        for i in range(sample_size):
            x = data['X'].iloc[i]
            y = data['Y'].iloc[i]
            z = data['Z'].iloc[i]

            # Y should be 2X + 1
            self.assertAlmostEqual(y, 2 * x + 1)

            # Z should be X + Y
            self.assertAlmostEqual(z, x + y)

    def test_get_observational_data(self):
        """Test that get_observational_data returns observational data correctly."""
        # Generate observational data
        sample_size = 100
        data = self.scm.get_observational_data(sample_size)

        # Check that the returned data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)

        # Check that the DataFrame has the correct columns
        self.assertEqual(set(data.columns), {'X', 'Y', 'Z'})

        # Check that the DataFrame has the correct number of rows
        self.assertEqual(len(data), sample_size)

        # Check that the relationships hold for each sample
        for i in range(sample_size):
            x = data['X'].iloc[i]
            y = data['Y'].iloc[i]
            z = data['Z'].iloc[i]

            # Y should be 2X + 1
            self.assertAlmostEqual(y, 2 * x + 1)

            # Z should be X + Y
            self.assertAlmostEqual(z, x + y)

    def test_sampling_with_intervention(self):
        """Test that sample_data works correctly with interventions."""
        # Perform an intervention
        self.scm.do_intervention('X', 3)

        # Generate samples with the intervention
        sample_size = 100
        data = self.scm.sample_data(sample_size)

        # Check that X is fixed to the intervention value
        for i in range(sample_size):
            self.assertEqual(data['X'].iloc[i], 3)

            # Check that other relationships still hold
            y = data['Y'].iloc[i]
            z = data['Z'].iloc[i]

            # Y should be 2X + 1 = 2*3 + 1 = 7
            self.assertAlmostEqual(y, 7)

            # Z should be X + Y = 3 + 7 = 10
            self.assertAlmostEqual(z, 10)

    def test_sample_interventional_data(self):
        """Test that sample_interventional_data generates interventional data correctly."""
        # Generate interventional data
        sample_size = 100
        interventions = {'X': 2}
        data = self.scm.sample_interventional_data(interventions, sample_size)

        # Check that X is fixed to the intervention value
        for i in range(sample_size):
            self.assertEqual(data['X'].iloc[i], 2)

            # Check that other relationships still hold
            y = data['Y'].iloc[i]
            z = data['Z'].iloc[i]

            # Y should be 2X + 1 = 2*2 + 1 = 5
            self.assertAlmostEqual(y, 5)

            # Z should be X + Y = 2 + 5 = 7
            self.assertAlmostEqual(z, 7)

        # Check that the original state is restored (no interventions)
        self.assertEqual(self.scm._interventions, {})

    def test_sampling_with_multiple_interventions(self):
        """Test that sample_data works correctly with multiple interventions."""
        # Perform multiple interventions
        interventions = {'X': 2, 'Y': 5}
        self.scm.multiple_interventions(interventions)

        # Generate samples with the interventions
        sample_size = 100
        data = self.scm.sample_data(sample_size)

        # Check that X and Y are fixed to their intervention values
        for i in range(sample_size):
            self.assertEqual(data['X'].iloc[i], 2)
            self.assertEqual(data['Y'].iloc[i], 5)

            # Check that Z relationship still holds
            z = data['Z'].iloc[i]

            # Z should be X + Y = 2 + 5 = 7
            self.assertAlmostEqual(z, 7)

    def test_sampling_with_random_seed(self):
        """Test that sample_data with the same random seed produces reproducible results."""
        # Generate two datasets with the same seed
        sample_size = 100
        data1 = self.scm.sample_data(sample_size, random_seed=42)
        data2 = self.scm.sample_data(sample_size, random_seed=42)

        # The datasets should be identical
        pd.testing.assert_frame_equal(data1, data2)

        # Generate a dataset with a different seed
        data3 = self.scm.sample_data(sample_size, random_seed=43)

        # The datasets should be different
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(data1, data3)

    def test_sampling_missing_structural_equation(self):
        """Test that sample_data raises an error when a variable doesn't have a structural equation."""
        # Create an SCM with a variable missing a structural equation
        scm = StructuralCausalModel()
        scm.add_variable('A')
        scm.add_variable('B')
        scm.define_causal_relationship('B', ['A'])

        # Define an equation for A but not for B
        def a_equation():
            return np.random.normal(0, 1)

        scm.define_structural_equation('A', a_equation)

        # Sampling should raise an error due to missing equation for B
        with self.assertRaises(ValueError):
            scm.sample_data(10)

    def test_sampling_with_probabilistic_equation(self):
        """Test that sample_data works correctly with probabilistic equations."""
        # Create an SCM with a probabilistic equation
        scm = StructuralCausalModel(random_state=42)
        scm.add_variable('A')
        scm.add_variable('B')
        scm.define_causal_relationship('B', ['A'])

        # Define a deterministic equation for A
        def a_equation():
            return 1.0

        # Define a probabilistic equation for B
        def b_equation(A, noise):
            return A + noise

        def noise_dist():
            return np.random.normal(0, 1)

        scm.define_structural_equation('A', a_equation)
        scm.define_probabilistic_equation('B', b_equation, noise_dist)

        # Generate samples
        sample_size = 100
        data = scm.sample_data(sample_size)

        # Check that A is always 1.0
        for i in range(sample_size):
            self.assertEqual(data['A'].iloc[i], 1.0)

        # B should be A + noise, so its mean should be close to 1.0
        self.assertAlmostEqual(data['B'].mean(), 1.0, delta=0.2)
        # The standard deviation should be close to 1.0
        self.assertAlmostEqual(data['B'].std(), 1.0, delta=0.2)


if __name__ == '__main__':
    unittest.main()
