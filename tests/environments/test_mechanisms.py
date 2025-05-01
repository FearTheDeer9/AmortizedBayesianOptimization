#!/usr/bin/env python3
"""
Unit tests for causal mechanisms.
"""

from causal_meta.environments.mechanisms import MechanismFactory
from typing import List  # For type hints
from causal_meta.environments.mechanisms import NonlinearMechanism
from typing import Dict, Any, Union  # For custom function type hint
import math
from causal_meta.environments.mechanisms import BaseMechanism, LinearMechanism
import unittest
import numpy as np
import os
import sys
from abc import ABCMeta

# Add project root to path to allow importing causal_meta
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


# A minimal concrete implementation for testing BaseMechanism

class MockMechanism(BaseMechanism):
    def compute(self, parent_values):
        # Simple sum of parent values
        return sum(parent_values.values()) if parent_values else 0


class TestBaseMechanism(unittest.TestCase):
    """Tests for the BaseMechanism abstract class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify that BaseMechanism cannot be instantiated directly."""
        with self.assertRaises(TypeError, msg="Should not be able to instantiate abstract BaseMechanism"):
            BaseMechanism()

    def test_mock_mechanism_instantiation(self):
        """Test instantiation of a concrete subclass."""
        mechanism = MockMechanism(noise_std=0.1, seed=42)
        self.assertIsInstance(mechanism, BaseMechanism)
        self.assertEqual(mechanism.noise_std, 0.1)
        self.assertEqual(mechanism.seed, 42)
        self.assertIsNotNone(mechanism._rng)

    def test_compute_call_deterministic(self):
        """Test calling the mechanism without noise."""
        mechanism = MockMechanism(noise_std=0.0, seed=1)
        parent_values = {'A': 2, 'B': 3}
        expected_value = 5  # 2 + 3
        computed_value = mechanism(parent_values)
        self.assertEqual(computed_value, expected_value)

        # Test with no parents
        computed_value_no_parents = mechanism({})
        self.assertEqual(computed_value_no_parents, 0)

    def test_compute_call_with_noise(self):
        """Test calling the mechanism with noise."""
        mechanism = MockMechanism(noise_std=1.0, seed=123)
        parent_values = {'X': 10}
        expected_deterministic_value = 10
        computed_value = mechanism(parent_values)

        # Check if noise was added (value should likely differ)
        self.assertNotEqual(computed_value, expected_deterministic_value,
                            "Value should differ from deterministic due to noise")

        # Check reproducibility with the same seed
        mechanism.set_seed(123)  # Reset RNG state
        computed_value_rerun = mechanism(parent_values)
        self.assertEqual(computed_value, computed_value_rerun,
                         "Calling with the same seed should produce the same noisy result")

    def test_repr(self):
        """Test the string representation."""
        mechanism = MockMechanism(noise_std=0.5, seed=99)
        expected_repr = "MockMechanism(noise_std=0.5, seed=99)"
        self.assertEqual(repr(mechanism), expected_repr)


class TestLinearMechanism(unittest.TestCase):
    """Tests for the LinearMechanism class."""

    def test_initialization(self):
        """Test initializing a LinearMechanism."""
        weights = {'A': 0.5, 'B': -1.0}
        mechanism = LinearMechanism(
            weights=weights, intercept=1.0, noise_std=0.1, seed=42)
        self.assertEqual(mechanism.weights, weights)
        self.assertEqual(mechanism.intercept, 1.0)
        self.assertEqual(mechanism.noise_std, 0.1)
        self.assertEqual(mechanism.seed, 42)
        self.assertEqual(mechanism._expected_parents, {'A', 'B'})

    def test_compute_deterministic(self):
        """Test deterministic computation (no noise)."""
        weights = {'X1': 2.0, 'X2': 3.0}
        mechanism = LinearMechanism(
            weights=weights, intercept=5.0, noise_std=0.0)
        parent_values = {'X1': 10, 'X2': -1}
        # Expected: 5.0 + (2.0 * 10) + (3.0 * -1) = 5.0 + 20.0 - 3.0 = 22.0
        expected_value = 22.0
        computed_value = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, expected_value)

    def test_compute_with_noise(self):
        """Test computation with noise."""
        weights = {'P1': 1.0}
        mechanism = LinearMechanism(
            weights=weights, intercept=0.0, noise_std=1.0, seed=123)
        parent_values = {'P1': 5}
        # Deterministic part is 5.0
        computed_value = mechanism(parent_values)
        self.assertNotAlmostEqual(computed_value, 5.0)

        # Check reproducibility
        mechanism.set_seed(123)
        computed_value_rerun = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, computed_value_rerun)

    def test_compute_no_parents(self):
        """Test computation with no parents (only intercept and noise)."""
        mechanism = LinearMechanism(weights={}, intercept=10.0, noise_std=0.0)
        computed_value = mechanism({})
        self.assertAlmostEqual(computed_value, 10.0)

        # With noise
        mechanism_noisy = LinearMechanism(
            weights={}, intercept=10.0, noise_std=1.0, seed=1)
        computed_noisy = mechanism_noisy({})
        self.assertNotAlmostEqual(computed_noisy, 10.0)

    def test_compute_raises_value_error_on_mismatch(self):
        """Test that compute raises error if parent keys don't match weights."""
        weights = {'A': 1.0, 'B': 1.0}
        mechanism = LinearMechanism(weights=weights)

        # Missing parent 'B'
        with self.assertRaises(ValueError):
            mechanism({'A': 5})

        # Extra parent 'C'
        with self.assertRaises(ValueError):
            mechanism({'A': 5, 'B': 2, 'C': 1})

        # Different parent 'D' instead of 'B'
        with self.assertRaises(ValueError):
            mechanism({'A': 5, 'D': 2})

    def test_repr_linear(self):
        """Test the string representation of LinearMechanism."""
        weights = {'A': 0.5, 'B': -1.0}
        mechanism = LinearMechanism(
            weights=weights, intercept=1.0, noise_std=0.1, seed=42)
        expected_repr = "LinearMechanism(weights={'A': 0.5, 'B': -1.0}, intercept=1.0, noise_std=0.1, seed=42)"
        self.assertEqual(repr(mechanism), expected_repr)

        # Test long repr
        long_weights = {f'P{i}': i for i in range(5)}
        mechanism_long = LinearMechanism(weights=long_weights)
        expected_long_repr = "LinearMechanism(weights=5 parents, intercept=0.0, noise_std=0.0, seed=None)"
        self.assertEqual(repr(mechanism_long), expected_long_repr)


# Import the new class


class TestNonlinearMechanism(unittest.TestCase):
    """Tests for the NonlinearMechanism class."""

    def test_initialization_predefined(self):
        """Test initializing with predefined function types."""
        weights = {'A': 1.0, 'B': 2.0}
        mech_quad = NonlinearMechanism(
            func_type='quadratic', weights=weights, noise_std=0.1, seed=1)
        self.assertEqual(mech_quad.func_type_name, 'quadratic')
        self.assertEqual(mech_quad.weights, weights)
        self.assertFalse(mech_quad._is_custom_dict_func)

        mech_sig = NonlinearMechanism(func_type='sigmoid', weights={'X': 1.0})
        self.assertEqual(mech_sig.func_type_name, 'sigmoid')
        self.assertFalse(mech_sig._is_custom_dict_func)

    def test_initialization_custom_callable(self):
        """Test initializing with a custom callable function."""
        def custom_func(lin_comb: float) -> float:
            return lin_comb ** 3
        weights = {'P': 1.0}
        mechanism = NonlinearMechanism(func_type=custom_func, weights=weights)
        self.assertEqual(mechanism.func_type_name, 'custom_func')
        self.assertEqual(mechanism.weights, weights)
        self.assertFalse(mechanism._is_custom_dict_func)

    def test_initialization_custom_callable_dict_input(self):
        """Test init with a custom callable expecting the parent dict."""
        def custom_dict_func(parents: Dict[Any, Union[float, int]]) -> float:
            return parents.get('A', 0) * parents.get('B', 0)
        # Weights are optional here if the function handles dict directly
        mechanism = NonlinearMechanism(
            func_type=custom_dict_func, weights=None)
        self.assertEqual(mechanism.func_type_name, 'custom_dict_func')
        self.assertEqual(mechanism.weights, {})  # Defaults to empty
        self.assertTrue(mechanism._is_custom_dict_func)

    def test_initialization_custom_string(self):
        """Test initializing with func_type='custom' and func in params."""
        def my_func(parents): return parents.get('A', 0) + 1
        params = {'custom_func': my_func}
        mech = NonlinearMechanism(func_type='custom', func_params=params)
        self.assertTrue(callable(mech.func))
        self.assertEqual(mech.func_type_name, 'custom')
        self.assertTrue(mech._is_custom_dict_func)

    def test_initialization_errors(self):
        """Test initialization errors."""
        with self.assertRaises(ValueError, msg="Weights required for predefined"):
            NonlinearMechanism(func_type='quadratic', weights=None)
        with self.assertRaises(ValueError, msg="Custom func missing in params"):
            NonlinearMechanism(func_type='custom', func_params={})
        with self.assertRaises(TypeError, msg="Invalid func_type"):
            NonlinearMechanism(func_type=123)

    def test_compute_predefined_quadratic(self):
        """Test computation with predefined quadratic function."""
        weights = {'A': 1.0, 'B': 2.0}  # A + 2B
        mechanism = NonlinearMechanism(
            func_type='quadratic', weights=weights, intercept=1.0, noise_std=0.0)
        parent_values = {'A': 3, 'B': 2}
        # Linear combination = 1.0 + (1.0 * 3) + (2.0 * 2) = 1.0 + 3 + 4 = 8.0
        # Quadratic func = 1.0 * (8.0 ** 2) = 64.0
        expected_value = 64.0
        computed_value = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, expected_value)

    def test_compute_predefined_sigmoid(self):
        """Test computation with predefined sigmoid function."""
        weights = {'X': 1.0}
        mechanism = NonlinearMechanism(
            func_type='sigmoid', weights=weights, intercept=0.0, noise_std=0.0)
        parent_values = {'X': 0}
        # Linear combination = 0.0 + (1.0 * 0) = 0.0
        # Sigmoid func = 1 / (1 + exp(-0)) = 1 / (1 + 1) = 0.5
        expected_value = 0.5
        computed_value = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, expected_value)

    def test_compute_custom_callable(self):
        """Test computation with a custom callable."""
        def custom_cube(lin_comb: float) -> float:
            return lin_comb ** 3
        weights = {'P': 2.0}
        mechanism = NonlinearMechanism(
            func_type=custom_cube, weights=weights, intercept=1.0, noise_std=0.0)
        parent_values = {'P': 2}
        # Linear combination = 1.0 + (2.0 * 2) = 5.0
        # Custom func = 5.0 ** 3 = 125.0
        expected_value = 125.0
        computed_value = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, expected_value)

    def test_compute_custom_callable_dict_input(self):
        """Test computation with custom callable expecting dict."""
        def custom_prod(parents: Dict[Any, Union[float, int]]) -> float:
            return parents.get('A', 1) * parents.get('B', 1)
        mechanism = NonlinearMechanism(
            func_type=custom_prod, weights=None, noise_std=0.0)
        parent_values = {'A': 5, 'B': 3}
        # Custom func = 5 * 3 = 15.0
        expected_value = 15.0
        computed_value = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, expected_value)

    def test_compute_with_noise(self):
        """Test nonlinear computation with noise."""
        weights = {'X': 1.0}
        mechanism = NonlinearMechanism(
            func_type='quadratic', weights=weights, noise_std=1.0, seed=321)
        parent_values = {'X': 3}
        # Deterministic: (1.0 * 3)**2 = 9.0
        computed_value = mechanism(parent_values)
        self.assertNotAlmostEqual(computed_value, 9.0)

        # Check reproducibility
        mechanism.set_seed(321)
        computed_value_rerun = mechanism(parent_values)
        self.assertAlmostEqual(computed_value, computed_value_rerun)

    def test_repr_nonlinear(self):
        """Test the string representation."""
        weights = {'A': 1.0}
        mechanism = NonlinearMechanism(
            func_type='sigmoid', weights=weights, noise_std=0.1, seed=5)
        expected_repr = "NonlinearMechanism(func_type=sigmoid, weights={'A': 1.0}, intercept=0.0, noise_std=0.1, seed=5)"
        self.assertEqual(repr(mechanism), expected_repr)

        def my_custom_func(p): return 1
        params = {'custom_func': my_custom_func}
        mech_custom = NonlinearMechanism(
            func_type='custom', func_params=params, seed=6)
        # Note: func_params might become complex, ensure repr handles it reasonably
        expected_custom_repr = f"NonlinearMechanism(func_type=custom, noise_std=0.0, func_params={{'custom_func': {repr(my_custom_func)}}}, seed=6)"
        self.assertEqual(repr(mech_custom), expected_custom_repr)


# Import the new class and dependencies
try:
    import torch
    import torch.nn as nn
    from causal_meta.environments.mechanisms import NeuralNetworkMechanism
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes if torch isn't installed for tests that might run

    class nn:
        Module = type('Module', (object,), {})
    NeuralNetworkMechanism = type('NeuralNetworkMechanism', (object,), {
                                  '__init__': lambda *args, **kwargs: None})


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed, skipping NeuralNetworkMechanism tests.")
class TestNeuralNetworkMechanism(unittest.TestCase):
    """Tests for the NeuralNetworkMechanism class."""

    def test_initialization_mlp(self):
        """Test initializing by specifying MLP layers."""
        parent_order = ['A', 'B']
        layers = [16, 8]  # Hidden layers
        activation = nn.ReLU
        mechanism = NeuralNetworkMechanism(
            input_parent_order=parent_order,
            mlp_layers=layers,
            activation=activation,
            noise_std=0.1,
            seed=42
        )
        self.assertIsInstance(mechanism.network, nn.Sequential)
        self.assertEqual(mechanism.input_parent_order, parent_order)
        self.assertEqual(mechanism.noise_std, 0.1)
        self.assertEqual(mechanism.seed, 42)
        # Check network structure (basic check)
        # Linear, ReLU, Linear, ReLU, Linear
        self.assertEqual(len(mechanism.network), 5)
        self.assertIsInstance(mechanism.network[0], nn.Linear)
        self.assertEqual(mechanism.network[0].in_features, 2)  # Input dim
        self.assertEqual(mechanism.network[0].out_features, 16)  # First hidden
        self.assertIsInstance(mechanism.network[-1], nn.Linear)
        self.assertEqual(mechanism.network[-1].in_features, 8)  # Last hidden
        self.assertEqual(mechanism.network[-1].out_features, 1)  # Output dim

    def test_initialization_prebuilt_network(self):
        """Test initializing with a pre-built network."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, x):
                return self.linear(x)

        network = SimpleNet()
        parent_order = ['X', 'Y']
        mechanism = NeuralNetworkMechanism(
            network=network, input_parent_order=parent_order, seed=1)
        self.assertIs(mechanism.network, network)
        self.assertEqual(mechanism.input_parent_order, parent_order)

    def test_initialization_errors(self):
        """Test initialization errors."""
        with self.assertRaises(ValueError, msg="Must provide network or mlp_layers"):
            NeuralNetworkMechanism(input_parent_order=['A'])
        with self.assertRaises(ValueError, msg="Cannot provide both network and mlp_layers"):
            NeuralNetworkMechanism(network=nn.Linear(1, 1), mlp_layers=[
                                   8], input_parent_order=['A'])
        with self.assertRaises(ValueError, msg="input_parent_order is required"):
            NeuralNetworkMechanism(mlp_layers=[8])

    def test_compute_mlp_deterministic(self):
        """Test computation with an MLP without noise."""
        parent_order = ['A', 'B']
        # Simple MLP: Input(2) -> Linear(1) -> Output(1)
        mechanism = NeuralNetworkMechanism(
            input_parent_order=parent_order,
            mlp_layers=[],  # No hidden layers
            noise_std=0.0,
            seed=1  # For weight initialization
        )
        # Manually set weights for predictable output (optional)
        # mechanism.network[0].weight.data.fill_(1.0)
        # mechanism.network[0].bias.data.fill_(0.5)
        # Input [2, 3] -> (1*2 + 1*3) + 0.5 = 5.5

        parent_values = {'A': 2.0, 'B': 3.0}
        computed_value = mechanism(parent_values)
        # Value depends on random init unless weights are set manually
        self.assertIsInstance(computed_value, float)

    def test_compute_with_noise(self):
        """Test computation with noise."""
        parent_order = ['X']
        mechanism = NeuralNetworkMechanism(
            input_parent_order=parent_order,
            mlp_layers=[4],  # Input(1) -> Linear(4) -> Linear(1)
            noise_std=1.0,
            seed=123
        )
        parent_values = {'X': 5.0}
        computed_value_no_noise = mechanism.compute(
            parent_values)  # Get deterministic part

        # Now call __call__ which adds noise
        computed_value_with_noise = mechanism(parent_values)

        self.assertNotAlmostEqual(
            computed_value_with_noise, computed_value_no_noise)

        # Check reproducibility
        mechanism.set_seed(123)  # Resets noise RNG and network init seed
        # Re-initialize network weights with seed
        mechanism._initialize_weights(mechanism.network)
        computed_value_rerun = mechanism(parent_values)
        self.assertAlmostEqual(computed_value_with_noise, computed_value_rerun)

    def test_compute_raises_value_error_on_mismatch(self):
        """Test compute raises error if parent keys don't match order."""
        parent_order = ['A', 'B']
        mechanism = NeuralNetworkMechanism(
            input_parent_order=parent_order, mlp_layers=[4])

        # Missing parent 'B'
        with self.assertRaises(ValueError):
            mechanism({'A': 5})

        # Extra parent 'C'
        with self.assertRaises(ValueError):
            mechanism({'A': 5, 'B': 2, 'C': 1})

    def test_repr_nn(self):
        """Test the string representation."""
        parent_order = ['A', 'B']
        mechanism = NeuralNetworkMechanism(
            input_parent_order=parent_order, mlp_layers=[8], seed=99)
        # Representation might vary slightly based on exact nn.Sequential repr
        expected_repr = "NeuralNetworkMechanism(network=Sequential, parents=['A', 'B'], output_dim=1, noise_std=0.0, seed=99)"
        self.assertEqual(repr(mechanism), expected_repr)


# Import the new factory


class TestMechanismFactory(unittest.TestCase):
    """Tests for the MechanismFactory class."""

    def test_create_linear(self):
        """Test creating a LinearMechanism via factory."""
        config = {
            'type': 'linear',
            'params': {
                'weights': {'A': 0.5, 'B': -1.0},
                'intercept': 1.0,
                'noise_std': 0.1,
                'seed': 42
            }
        }
        mechanism = MechanismFactory.create_mechanism(config)
        self.assertIsInstance(mechanism, LinearMechanism)
        self.assertEqual(mechanism.weights, {'A': 0.5, 'B': -1.0})
        self.assertEqual(mechanism.intercept, 1.0)
        self.assertEqual(mechanism.noise_std, 0.1)
        self.assertEqual(mechanism.seed, 42)

    def test_create_nonlinear_predefined(self):
        """Test creating a predefined NonlinearMechanism via factory."""
        config = {
            'type': 'nonlinear',
            'params': {
                'func_type': 'sigmoid',
                'weights': {'X': 1.0},
                'intercept': 0.0,
                'noise_std': 0.0,
                'seed': 5
            }
        }
        mechanism = MechanismFactory.create_mechanism(config)
        self.assertIsInstance(mechanism, NonlinearMechanism)
        self.assertEqual(mechanism.func_type_name, 'sigmoid')
        self.assertEqual(mechanism.weights, {'X': 1.0})
        self.assertEqual(mechanism.noise_std, 0.0)

    def test_create_nonlinear_custom_func(self):
        """Test creating a custom func NonlinearMechanism via factory."""
        def my_func(parents): return parents.get('A', 0)**2
        config = {
            'type': 'nonlinear',
            'params': {
                'func_type': 'custom',
                # Weights could be None or {} if func handles dict
                'weights': None,  # Or {}
                'func_params': {'custom_func': my_func},
                'noise_std': 0.0,
                'seed': 6
            }
        }
        mechanism = MechanismFactory.create_mechanism(config)
        self.assertIsInstance(mechanism, NonlinearMechanism)
        self.assertTrue(mechanism._is_custom_dict_func)
        self.assertEqual(mechanism({'A': 3}), 9.0)  # 3**2

    @unittest.skipIf(not TORCH_AVAILABLE, "PyTorch not installed, skipping NN mechanism factory test.")
    def test_create_neural_network(self):
        """Test creating a NeuralNetworkMechanism via factory."""
        config = {
            'type': 'neural_network',
            'params': {
                'input_parent_order': ['P1', 'P2'],
                'mlp_layers': [10],  # Single hidden layer
                'activation': nn.Tanh,  # Specify activation
                'noise_std': 0.05,
                'seed': 101
            }
        }
        mechanism = MechanismFactory.create_mechanism(config)
        self.assertIsInstance(mechanism, NeuralNetworkMechanism)
        self.assertEqual(mechanism.input_parent_order, ['P1', 'P2'])
        self.assertIsInstance(mechanism.network, nn.Sequential)
        self.assertEqual(mechanism.noise_std, 0.05)
        self.assertEqual(mechanism.seed, 101)

    def test_create_custom_func_wrapper(self):
        """Test creating a mechanism from a simple function using 'custom_func' type."""
        def simple_sum(parents): return sum(parents.values())
        config = {
            'type': 'custom_func',
            'params': {
                'function': simple_sum,
                'noise_std': 0.1,
                'seed': 7
            }
        }
        mechanism = MechanismFactory.create_mechanism(config)
        # Check it's a BaseMechanism, specific type might be internal wrapper
        self.assertIsInstance(mechanism, BaseMechanism)
        self.assertEqual(mechanism.noise_std, 0.1)
        self.assertEqual(mechanism.seed, 7)
        # Test computation
        self.assertNotAlmostEqual(
            mechanism({'A': 2, 'B': 3}), 5.0)  # Should have noise

    def test_factory_errors(self):
        """Test error handling in the factory."""
        with self.assertRaises(ValueError, msg="Missing type"):
            MechanismFactory.create_mechanism({'params': {}})
        with self.assertRaises(ValueError, msg="Unknown type"):
            MechanismFactory.create_mechanism(
                {'type': 'unknown', 'params': {}})
        with self.assertRaises(ValueError, msg="Missing weights for linear"):
            MechanismFactory.create_mechanism({'type': 'linear', 'params': {}})
        with self.assertRaises(ValueError, msg="Missing func_type for nonlinear"):
            MechanismFactory.create_mechanism(
                {'type': 'nonlinear', 'params': {}})
        with self.assertRaises(ValueError, msg="Missing function for custom_func"):
            MechanismFactory.create_mechanism(
                {'type': 'custom_func', 'params': {}})

    def test_generate_random_mechanism(self):
        """Test the utility for generating random mechanisms."""
        # Test linear
        mech_lin = MechanismFactory.generate_random_mechanism(
            mechanism_type='linear', parent_ids=['X', 'Y'], seed=10)
        self.assertIsInstance(mech_lin, LinearMechanism)
        self.assertEqual(set(mech_lin.weights.keys()), {'X', 'Y'})
        self.assertTrue(0.0 <= mech_lin.noise_std <= 0.2)

        # Test nonlinear (quadratic example)
        mech_nl = MechanismFactory.generate_random_mechanism(
            mechanism_type='nonlinear', parent_ids=['Z'], seed=11)
        self.assertIsInstance(mech_nl, NonlinearMechanism)
        self.assertEqual(set(mech_nl.weights.keys()), {'Z'})
        self.assertTrue(0.0 <= mech_nl.noise_std <= 0.2)
        self.assertEqual(mech_nl.func_type_name, 'quadratic')

        # Test unknown type
        with self.assertRaises(ValueError):
            MechanismFactory.generate_random_mechanism(
                mechanism_type='unknown')


if __name__ == "__main__":
    # Ensure the project root is in the path when running directly
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    unittest.main()
