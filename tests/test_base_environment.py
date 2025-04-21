"""
Unit tests for the BaseEnvironment interface.

This module contains tests for the BaseEnvironment abstract class,
using a simple concrete implementation to verify the interface.
"""
import unittest
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union

from causal_meta.environments.base import BaseEnvironment
from causal_meta.graph.causal_graph import CausalGraph


class SimpleEnvironment(BaseEnvironment):
    """
    A simple concrete implementation of BaseEnvironment for testing.

    This class implements the abstract methods of BaseEnvironment
    with minimal functionality for testing purposes.
    """

    def __init__(self, **kwargs):
        """Initialize a simple environment for testing."""
        super().__init__(**kwargs)

        # Create a simple causal graph if one wasn't provided
        if self._causal_graph is None:
            self._causal_graph = CausalGraph()
            self._causal_graph.add_node('X')
            self._causal_graph.add_node('Y')
            self._causal_graph.add_node('Z')
            self._causal_graph.add_edge('X', 'Y')
            self._causal_graph.add_edge('Z', 'Y')

        # Set variable names if not provided
        if not self._variable_names:
            self._variable_names = ['X', 'Y', 'Z']

    def sample_data(self, sample_size: int,
                    random_seed: Optional[int] = None) -> pd.DataFrame:
        """Sample data from this simple environment."""
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate data based on a simple linear model
        X = np.random.normal(0, 1, sample_size)
        Z = np.random.normal(0, 1, sample_size)

        # Apply any interventions
        if 'X' in self._interventions:
            X = np.ones(sample_size) * self._interventions['X']
        if 'Z' in self._interventions:
            Z = np.ones(sample_size) * self._interventions['Z']

        # Y depends on X and Z
        Y = 0.5 * X + 0.5 * Z + np.random.normal(0, 0.1, sample_size)

        # Create a DataFrame
        data = pd.DataFrame({
            'X': X,
            'Y': Y,
            'Z': Z
        })

        return data

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        self._interventions = {}

    def get_observational_data(self, sample_size: int,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
        """Get purely observational data from the environment."""
        # Save current interventions
        saved_interventions = self._interventions.copy()

        # Reset interventions
        self._interventions = {}

        # Sample data
        data = self.sample_data(sample_size, random_seed)

        # Restore interventions
        self._interventions = saved_interventions

        return data

    def do_intervention(self, target: Any, value: Any) -> None:
        """Perform a 'do' intervention on the environment."""
        if target not in self._variable_names:
            raise ValueError(f"Target variable {target} does not exist")

        self._interventions[target] = value

    def sample_interventional_data(self, interventions: Dict[Any, Any],
                                   sample_size: int,
                                   random_seed: Optional[int] = None) -> pd.DataFrame:
        """Sample data from the environment under specified interventions."""
        # Save current interventions
        saved_interventions = self._interventions.copy()

        # Apply new interventions
        for target, value in interventions.items():
            self.do_intervention(target, value)

        # Sample data
        data = self.sample_data(sample_size, random_seed)

        # Restore original interventions
        self._interventions = saved_interventions

        return data

    def multiple_interventions(self, interventions: Dict[Any, Any]) -> None:
        """Perform multiple interventions on the environment simultaneously."""
        for target, value in interventions.items():
            if target not in self._variable_names:
                raise ValueError(f"Target variable {target} does not exist")

            self._interventions[target] = value

    def get_intervention_effects(self, target: Any, outcome: Any,
                                 intervention_values: List[Any],
                                 sample_size: int = 1000,
                                 random_seed: Optional[int] = None) -> Dict[Any, float]:
        """Calculate the effects of different interventions on an outcome."""
        if target not in self._variable_names:
            raise ValueError(f"Target variable {target} does not exist")
        if outcome not in self._variable_names:
            raise ValueError(f"Outcome variable {outcome} does not exist")

        effects = {}

        for value in intervention_values:
            # Sample data under the intervention
            data = self.sample_interventional_data(
                {target: value}, sample_size, random_seed)

            # Calculate the mean outcome
            effects[value] = data[outcome].mean()

        return effects

    def evaluate_counterfactual(self, factual_data: pd.DataFrame,
                                interventions: Dict[Any, Any]) -> pd.DataFrame:
        """Evaluate counterfactual outcomes given factual data and interventions."""
        # This is a very simplified implementation
        # In a real implementation, we would need to abduction, action, prediction

        counterfactual_data = factual_data.copy()

        # For simplicity, we just use the factual values and modify Y
        for _, row in counterfactual_data.iterrows():
            # Set intervention values
            for target, value in interventions.items():
                row[target] = value

            # Recalculate Y based on the model
            if 'Y' in self._variable_names:
                row['Y'] = 0.5 * row['X'] + 0.5 * \
                    row['Z'] + np.random.normal(0, 0.1)

        return counterfactual_data

    def predict_outcome(self, interventions: Dict[Any, Any],
                        conditions: Optional[Dict[Any, Any]] = None) -> Dict[str, float]:
        """Predict the expected outcome of variables given interventions and conditions."""
        if conditions is None:
            conditions = {}

        # For simplicity, we'll just predict the mean outcomes
        # In a real implementation, this would be more sophisticated

        # Ensure all variables exist
        for var in list(interventions.keys()) + list(conditions.keys()):
            if var not in self._variable_names:
                raise ValueError(f"Variable {var} does not exist")

        # Default values for variables not specified
        X = 0.0
        Z = 0.0

        # Apply conditions
        if 'X' in conditions:
            X = conditions['X']
        if 'Z' in conditions:
            Z = conditions['Z']

        # Apply interventions (overriding conditions)
        if 'X' in interventions:
            X = interventions['X']
        if 'Z' in interventions:
            Z = interventions['Z']

        # Calculate expected Y
        Y = 0.5 * X + 0.5 * Z

        return {'X': X, 'Y': Y, 'Z': Z}

    def compute_effect(self, treatment: Any, outcome: Any,
                       treatment_value: Any, baseline_value: Optional[Any] = None,
                       sample_size: int = 1000,
                       random_seed: Optional[int] = None) -> float:
        """Compute the causal effect of a treatment on an outcome."""
        if treatment not in self._variable_names:
            raise ValueError(f"Treatment variable {treatment} does not exist")
        if outcome not in self._variable_names:
            raise ValueError(f"Outcome variable {outcome} does not exist")

        # Default baseline value is 0
        if baseline_value is None:
            baseline_value = 0.0

        # Sample data under treatment and baseline
        treatment_data = self.sample_interventional_data(
            {treatment: treatment_value}, sample_size // 2, random_seed)
        baseline_data = self.sample_interventional_data(
            {treatment: baseline_value}, sample_size // 2, random_seed)

        # Calculate treatment effect
        treatment_outcome = treatment_data[outcome].mean()
        baseline_outcome = baseline_data[outcome].mean()

        return treatment_outcome - baseline_outcome

    def counterfactual_distribution(self, factual_data: pd.DataFrame,
                                    interventions: Dict[Any, Any],
                                    target_variables: Optional[List[Any]] = None,
                                    num_samples: int = 100,
                                    random_seed: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """Compute the distribution of counterfactual outcomes."""
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Default to all variables if not specified
        if target_variables is None:
            target_variables = self._variable_names

        # Validate variables
        for var in list(interventions.keys()) + target_variables:
            if var not in self._variable_names:
                raise ValueError(f"Variable {var} does not exist")

        # Generate samples
        counterfactual_samples = {}

        for var in target_variables:
            if var in interventions:
                # If the variable is directly intervened upon, its value is fixed
                counterfactual_samples[var] = np.ones(
                    num_samples) * interventions[var]
            elif var == 'Y':
                # Y depends on X and Z
                X = np.ones(num_samples) * (
                    interventions.get('X', factual_data['X'].iloc[0]))
                Z = np.ones(num_samples) * (
                    interventions.get('Z', factual_data['Z'].iloc[0]))
                Y = 0.5 * X + 0.5 * Z + np.random.normal(0, 0.1, num_samples)
                counterfactual_samples[var] = Y
            else:
                # For other variables, use factual values if not intervened upon
                counterfactual_samples[var] = np.ones(
                    num_samples) * factual_data[var].iloc[0]

        return counterfactual_samples


class TestBaseEnvironment(unittest.TestCase):
    """Tests for the BaseEnvironment interface using SimpleEnvironment."""

    def setUp(self):
        """Set up for each test."""
        self.env = SimpleEnvironment()

    def test_initialization(self):
        """Test environment initialization."""
        # Check that the environment has the expected attributes
        self.assertIsNotNone(self.env._causal_graph)
        self.assertEqual(self.env._variable_names, ['X', 'Y', 'Z'])
        self.assertEqual(self.env._interventions, {})

    def test_sample_data(self):
        """Test sampling data from the environment."""
        data = self.env.sample_data(10, random_seed=42)

        # Check that the data has the expected shape and columns
        self.assertEqual(data.shape, (10, 3))
        self.assertTrue(all(col in data.columns for col in ['X', 'Y', 'Z']))

    def test_interventions(self):
        """Test performing interventions."""
        # Perform an intervention
        self.env.do_intervention('X', 5.0)

        # Check that the intervention was recorded
        self.assertEqual(self.env._interventions, {'X': 5.0})

        # Sample data and check that X is fixed at 5.0
        data = self.env.sample_data(10)
        self.assertTrue(all(data['X'] == 5.0))

        # Reset and check that interventions are cleared
        self.env.reset()
        self.assertEqual(self.env._interventions, {})

    def test_multiple_interventions(self):
        """Test performing multiple interventions."""
        # Perform multiple interventions
        self.env.multiple_interventions({'X': 5.0, 'Z': -2.0})

        # Check that the interventions were recorded
        self.assertEqual(self.env._interventions, {'X': 5.0, 'Z': -2.0})

        # Sample data and check that X and Z are fixed
        data = self.env.sample_data(10)
        self.assertTrue(all(data['X'] == 5.0))
        self.assertTrue(all(data['Z'] == -2.0))

    def test_observational_data(self):
        """Test getting observational data."""
        # Set an intervention
        self.env.do_intervention('X', 5.0)

        # Get observational data and check that X is not fixed
        data = self.env.get_observational_data(10, random_seed=42)
        self.assertFalse(all(data['X'] == 5.0))

        # Check that the intervention is still active
        self.assertEqual(self.env._interventions, {'X': 5.0})

    def test_sample_interventional_data(self):
        """Test sampling interventional data."""
        # Set an initial intervention
        self.env.do_intervention('X', 5.0)

        # Sample data with a different intervention
        data = self.env.sample_interventional_data(
            {'Z': -2.0}, 10, random_seed=42)

        # Check that Z is fixed in the data
        self.assertTrue(all(data['Z'] == -2.0))

        # Check that the original intervention is still active
        self.assertEqual(self.env._interventions, {'X': 5.0})

    def test_intervention_effects(self):
        """Test calculating intervention effects."""
        # Calculate effects of X on Y for different values
        effects = self.env.get_intervention_effects('X', 'Y', [0.0, 1.0, 2.0],
                                                    sample_size=100, random_seed=42)

        # Check that the effects dictionary has the expected keys
        self.assertEqual(set(effects.keys()), {0.0, 1.0, 2.0})

        # Check that higher X values lead to higher Y values
        self.assertLess(effects[0.0], effects[1.0])
        self.assertLess(effects[1.0], effects[2.0])

    def test_utility_methods(self):
        """Test utility methods."""
        # Test getting the causal graph
        graph = self.env.get_causal_graph()
        self.assertIsNotNone(graph)

        # Test getting variable names
        var_names = self.env.get_variable_names()
        self.assertEqual(var_names, ['X', 'Y', 'Z'])

        # Test getting variable info
        var_info = self.env.get_variable_info()
        self.assertEqual(set(var_info.keys()), {'X', 'Y', 'Z'})
        self.assertEqual(var_info['X']['children'], ['Y'])
        self.assertEqual(var_info['Y']['parents'], ['X', 'Z'])

        # Test validation
        self.assertTrue(self.env.validate())

    def test_compute_effect(self):
        """Test computing causal effects."""
        # Compute effect of X on Y
        effect = self.env.compute_effect('X', 'Y', 1.0, 0.0,
                                         sample_size=100, random_seed=42)

        # The effect should be approximately 0.5 (from the model Y = 0.5X + 0.5Z)
        self.assertAlmostEqual(effect, 0.5, delta=0.1)

    def test_counterfactual_distribution(self):
        """Test computing counterfactual distributions."""
        # Create some factual data
        factual_data = pd.DataFrame({'X': [0.0], 'Y': [0.0], 'Z': [0.0]})

        # Compute counterfactual distribution for Y when X=1.0
        cf_dist = self.env.counterfactual_distribution(
            factual_data, {'X': 1.0}, ['Y'], num_samples=100, random_seed=42)

        # Check that the distribution has the expected keys and shape
        self.assertEqual(set(cf_dist.keys()), {'Y'})
        self.assertEqual(cf_dist['Y'].shape, (100,))

        # The mean Y should be approximately 0.5 (from Y = 0.5X + 0.5Z with X=1, Z=0)
        self.assertAlmostEqual(cf_dist['Y'].mean(), 0.5, delta=0.1)


if __name__ == '__main__':
    unittest.main()
